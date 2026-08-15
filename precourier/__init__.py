"""PreCourier — pre-dispatch RTO prevention for Indian COD commerce.

The core of this package imports nothing outside the Python standard library.
That is a hard requirement, not an aesthetic: PRD section 9 promises that

    python -m precourier.backtest --seed 42

reproduces every published number on a machine with nothing installed, and
`tests/test_stdlib_only.py` asserts it. FastAPI and Jinja2 are confined to
`precourier.api`, `precourier.console` and `precourier.storefront`. Network and
ML libraries are imported at call time, never at module import.
"""

__version__ = "0.1.0"
