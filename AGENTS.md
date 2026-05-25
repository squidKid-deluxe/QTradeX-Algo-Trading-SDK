# QTradeX — Agent Guide

## Build & Install
- Dev install: `pip install -e .` (requires Cython>=3)
- Build wheel: `python -m build --wheel --outdir dist`
- **numpy<2** constraint in `pyproject.toml`

## Codegen
Regenerate indicator wrappers when tulipy adds/fixes functions:
`python qtradex/indicators/codegen.py`
This rewrites `qtradex/indicators/tulipy_wrapped.py` and the subpackage `__init__.py`.

## Cython
- `qtradex/indicators/utilities.pyx` auto-compiles via setuptools.
- `utilities.c` and `qi.c` are gitignored — built fresh during `pip install -e .`
- The sole `.pyx` source is `utilities.pyx`. `qi.py` is pure Python.

## Testing
- No formal test suite or CI tests.
- Only `qtradex/indicators/candle_class_tests.py` exists (uses `talib`, which is **not** a dependency — will fail unless it's installed separately).
- Run manually: `python qtradex/indicators/candle_class_tests.py`

## Linting & Formatting
No linting/formatting tooling is configured. Don't add or enforce any.

## CI
Only a release-publish workflow (`.github/workflows/publish.yml`). No test/lint gate.

## Architecture
- Single package `qtradex`, not a monorepo.
- Entrypoint: `qtradex.__init__` re-exports the public API.
- CLI entry: `qtradex-tune-manager` → `qtradex.core.tune_manager:main`
- Bot base class: `qtradex.core.BaseBot` — override `indicators()`, `strategy()`, `fitness()`, `plot()`.
- Dispatch: `qx.core.dispatch(bot, data, wallet)` — interactive CLI menu (backtest / optimize / papertrade / live).
- Data: `qtradex.public.Data(exchange, asset, currency, begin, end)` — fetches candles via CCXT, yfinance, etc.
- Optimizers: QPSO, LSGA, IPSE, AION in `qtradex.optimizers`.
- IPC/candle cache: `common/pipe/` (gitignored) — JSON file-based concurrent IPC and cached candle data.
- Tune caching: per-bot `tunes/` directory next to the bot's source file (`demos/tunes/` is gitignored).

## Platform
Linux-only (classifier: `Operating System :: POSIX :: Linux`).

## Signals
`Buy(price, maxvolume)`, `Sell(price, maxvolume)`, `Thresholds(buying, selling)`, `Hold`.
Signal `is_override = True` means the signal is acted on immediately (not queued for limit orders).
