"""Connection handling, schema application, and the tenant-scoping guard.

Three responsibilities, in ascending order of how much they matter:

1.  Open SQLite with settings that make it behave like a real database --
    foreign keys enforced, WAL journalling, a busy timeout, and `IMMEDIATE`
    transactions so two concurrent writers fail fast instead of deadlocking.

2.  Apply `schema.sql` idempotently, and record the version.

3.  Refuse to open the ground-truth database. PRD section 9 keeps true outcomes
    in a separate file that only the backtest may read, and section 13 calls
    leakage from it "the most dangerous failure mode in this build". A test walks
    the import graph to prove the agent's modules cannot reach the truth reader
    at all; the check in `connect` is the second lock on the same door, for the
    case where someone passes a path at runtime.

Queries are written with `:name` placeholders, which SQLite supports natively and
which survive a port to Postgres with a paramstyle shim. Positional `?` would
not: at twenty-odd columns, a silently shifted tuple is the kind of bug that
writes a merchant's freight into their margin.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "schema.sql")

#: The public schema version remains stable for additive Stage 1 tables. The one
#: additive column below is applied explicitly for databases created by Stage 0;
#: this keeps an existing merchant database usable without a destructive rewrite.
SCHEMA_VERSION = 1

#: Filenames the agent's runtime is not allowed to open. See module docstring.
FORBIDDEN_BASENAMES = frozenset({"truth.db", "truth.sqlite", "ground_truth.db"})


class StorageError(RuntimeError):
    """A storage-layer failure the caller is expected to handle."""


class GroundTruthLeak(StorageError):
    """An attempt to open the ground-truth database from the agent's runtime.

    Raised rather than logged. A silent fallback here would produce a backtest
    whose numbers look excellent and mean nothing, which is worse than a crash.
    """


def _reject_truth_path(path: str) -> None:
    base = os.path.basename(path).lower()
    if base in FORBIDDEN_BASENAMES:
        raise GroundTruthLeak(
            "refusing to open %r from the agent runtime. True outcomes are "
            "readable only by precourier.worldgen._truth, which the backtest "
            "imports directly. If you need labels for training, read the "
            "`outcomes` table -- those are the outcomes the merchant actually "
            "reported, which is all the agent is ever allowed to learn from."
            % (path,))


class Database:
    """An open SQLite connection, plus the query helpers the repos use.

    Not a connection pool. SQLite in WAL mode supports one writer and many
    readers, and the API layer holds one `Database` per process, which is the
    correct shape for both the demo and a single-node deployment. The seam for a
    pool is `Database.connect`, and nothing above this class assumes otherwise.
    """

    def __init__(self, conn: sqlite3.Connection, path: str) -> None:
        self._conn = conn
        self.path = path

    # ---------------------------------------------------------------- opening

    @classmethod
    def connect(cls, path: str = ":memory:", *, apply_schema: bool = True,
                read_only: bool = False) -> "Database":
        """Open `path`, configure it, and bring the schema up to date."""
        _reject_truth_path(path)

        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)

        # `check_same_thread=False` because the API serves sync endpoints from a
        # threadpool, so one connection is touched by several worker threads. It is
        # safe only because every request holds `api.deps.AppState.lock` for the
        # duration of its database work -- the flag alone would be a data race.
        # That lock is also this build's real concurrency ceiling: one writer at a
        # time. It is the correct trade for a single-file SQLite deployment, and
        # the Postgres port (everything SQL lives in `store/`) is the answer if the
        # ceiling ever matters.
        if read_only and path != ":memory:":
            uri = "file:%s?mode=ro" % path
            conn = sqlite3.connect(uri, uri=True, timeout=10.0,
                                   isolation_level=None, check_same_thread=False)
        else:
            conn = sqlite3.connect(path, timeout=10.0, isolation_level=None,
                                   check_same_thread=False)

        conn.row_factory = sqlite3.Row

        # Referential integrity is off by default in SQLite. Leaving it off would
        # let an order reference a merchant that does not exist, which is exactly
        # the tenant-isolation failure this schema is meant to prevent.
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        if not read_only:
            # WAL lets the console read while the API writes. On :memory: it is a
            # no-op, so this is unconditional rather than guarded.
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")

        db = cls(conn, path)
        if apply_schema and not read_only:
            db.apply_schema()
        return db

    def apply_schema(self) -> None:
        """Run `schema.sql`. Idempotent: every statement is `IF NOT EXISTS`."""
        with open(SCHEMA_PATH, encoding="utf-8") as handle:
            ddl = handle.read()
        # executescript() commits any open transaction, so it runs outside `tx`.
        self._conn.executescript(ddl)
        self._apply_additive_migrations()
        row = self._conn.execute(
            "SELECT MAX(version) AS v FROM schema_version").fetchone()
        if row is None or row["v"] is None:
            self._conn.execute(
                "INSERT INTO schema_version (version, applied_at) "
                "VALUES (:v, datetime('now'))", {"v": SCHEMA_VERSION})
        elif int(row["v"]) > SCHEMA_VERSION:
            raise StorageError(
                "database at %s was written by schema version %d; this build "
                "understands %d. Refusing to run against a newer schema."
                % (self.path, int(row["v"]), SCHEMA_VERSION))

    def _apply_additive_migrations(self) -> None:
        """Bring columns added after the initial schema into old SQLite files."""
        columns = {str(row["name"]) for row in self._conn.execute(
            "PRAGMA table_info(merchant_config)")}
        if "holdout_rate_percent" not in columns:
            self._conn.execute(
                "ALTER TABLE merchant_config ADD COLUMN holdout_rate_percent "
                "INTEGER NOT NULL DEFAULT 10 CHECK (holdout_rate_percent BETWEEN 0 AND 100)")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----------------------------------------------------------- transactions

    @contextlib.contextmanager
    def tx(self) -> Iterator["Database"]:
        """A write transaction. Commits on clean exit, rolls back on any raise.

        `BEGIN IMMEDIATE` takes the write lock up front. The default deferred
        begin acquires it on first write, which under concurrency can surface as
        `SQLITE_BUSY` partway through a multi-statement decision -- half an
        order's reasoning committed and half not.

        Reentrant: a nested `tx()` joins the outer transaction rather than
        starting a second one, so a repo method that writes an audit row can be
        called either standalone or inside a larger unit of work.
        """
        if self._conn.in_transaction:
            yield self
            return
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    # ----------------------------------------------------------------- access

    def execute(self, sql: str, params: Optional[Dict[str, Any]] = None
                ) -> sqlite3.Cursor:
        return self._conn.execute(sql, params or {})

    def executemany(self, sql: str,
                    rows: Iterable[Dict[str, Any]]) -> sqlite3.Cursor:
        return self._conn.executemany(sql, list(rows))

    def one(self, sql: str, params: Optional[Dict[str, Any]] = None
            ) -> Optional[sqlite3.Row]:
        return self._conn.execute(sql, params or {}).fetchone()

    def all(self, sql: str, params: Optional[Dict[str, Any]] = None
            ) -> List[sqlite3.Row]:
        return list(self._conn.execute(sql, params or {}).fetchall())

    def scalar(self, sql: str, params: Optional[Dict[str, Any]] = None,
               default: Any = None) -> Any:
        row = self.one(sql, params)
        if row is None:
            return default
        value = row[0]
        return default if value is None else value

    def insert(self, table: str, values: Dict[str, Any]) -> None:
        """Insert one row. Column names come from `values`, never from a caller
        string, so this cannot be turned into an injection point."""
        columns = sorted(values)
        sql = "INSERT INTO %s (%s) VALUES (%s)" % (
            _safe_identifier(table),
            ", ".join(_safe_identifier(c) for c in columns),
            ", ".join(":" + c for c in columns))
        self._conn.execute(sql, values)

    def exists(self, sql: str, params: Optional[Dict[str, Any]] = None) -> bool:
        return self.one(sql, params) is not None

    @property
    def raw(self) -> sqlite3.Connection:
        """Escape hatch for `executescript` and integrity checks. Used sparingly."""
        return self._conn

    # ------------------------------------------------------------ diagnostics

    def integrity_check(self) -> List[str]:
        """SQLite's own consistency check, plus foreign-key violations.

        Surfaced on the console's health page: a merchant deciding whether to
        trust this thing with their dispatch queue is entitled to see it.
        """
        problems = [row[0] for row in self._conn.execute("PRAGMA integrity_check")
                    if row[0] != "ok"]
        for row in self._conn.execute("PRAGMA foreign_key_check"):
            problems.append("foreign key violation: %s" % (tuple(row),))
        return problems

    def table_names(self) -> List[str]:
        return [row[0] for row in self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def _safe_identifier(name: str) -> str:
    """Allow only identifiers this codebase actually uses.

    `insert` interpolates table and column names, which parameters cannot carry.
    Every such name is a literal in our own source, so this is a tripwire against
    a future refactor that starts passing them in from a request rather than a
    defence against today's callers.
    """
    if not name or not all(ch.isalnum() or ch == "_" for ch in name):
        raise StorageError("unsafe SQL identifier %r" % (name,))
    return name


def rows_to_dicts(rows: Sequence[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]
