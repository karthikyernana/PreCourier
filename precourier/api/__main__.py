"""Entry point for `python -m precourier.api`.

    python -m precourier.api                 # start the server
    python -m precourier.api --sweep         # release expired holds and exit
    python -m precourier.api --host 0.0.0.0 # bind to all interfaces

The server reads PRECOURIER_DB (default: var/precourier.db) and
PRECOURIER_PEPPER from the environment. See README for a full walkthrough.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional, Sequence


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m precourier.api",
        description="PreCourier API server and hold-sweep utility")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="port to bind (default: 8000)")
    parser.add_argument("--sweep", action="store_true",
                        help="release every expired hold then exit (no server)")
    parser.add_argument("--log-level", default="info",
                        choices=("debug", "info", "warning", "error"),
                        help="log level (default: info)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ")

    from . import AppState, create_app, sweep_all
    from ..audit import AuditLog
    from ..runtime import Runtime
    from ..store.db import Database

    runtime = Runtime.live(pepper=os.environ.get("PRECOURIER_PEPPER", ""))
    db = Database.connect(os.environ.get("PRECOURIER_DB", "var/precourier.db"))
    audit = AuditLog(db, runtime)

    if args.sweep:
        app_state = AppState(db=db, runtime=runtime, audit=audit)
        released = sweep_all(app_state)
        print("released %d order(s)" % released)
        return 0

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed. Install it with:\n"
            "  pip install uvicorn[standard]\n"
            "or run the server via a different ASGI host.",
            file=sys.stderr)
        return 1

    app = create_app(db=db, runtime=runtime)
    uvicorn.run(app, host=args.host, port=args.port,
                log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
