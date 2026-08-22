"""Storage: schema, connection handling, and the tenant-scoped repositories.

`schema.sql` is the source of truth for shape and mutability. `db.py` opens it.
`repo.py` is the only module that writes SQL against it -- everything above the
store talks to repositories, so tenant scoping lives in one auditable place
instead of being re-remembered at each call site.
"""

from .db import (Database, GroundTruthLeak, SCHEMA_VERSION, StorageError,
                 rows_to_dicts)

__all__ = ["Database", "GroundTruthLeak", "SCHEMA_VERSION", "StorageError",
           "rows_to_dicts"]
