# QTradeX — Agent Guide

## Build & Install
- Dev install: `pip install -e .` (requires Cython>=3, `numpy<2`)
- Build wheel: `python -m build --wheel --outdir dist`

## Codegen
Regenerate indicator wrappers when tulipy adds/fixes functions:
`python qtradex/indicators/codegen.py`
Rewrites `qtradex/indicators/tulipy_wrapped.py` and the subpackage `__init__.py`.

## Cython
- `qtradex/indicators/utilities.pyx` is the sole `.pyx` — auto-compiled via setuptools.
- `utilities.c` generated from `.pyx`; `qi.py` is pure Python (no `.pyx`).
- Built artifacts (`*.c*`, `*.so`) are gitignored — rebult during `pip install -e .`

## Testing
- No formal test suite or CI tests.
- Only `qtradex/indicators/candle_class_tests.py` exists (needs `talib` — **not** a dependency; will fail unless installed separately).
- Run manually: `python qtradex/indicators/candle_class_tests.py`

## Linting & Formatting
None configured. Don't add any.

## CI
Only release-publish (`.github/workflows/publish.yml`). No test/lint gate.

## Architecture
- Single package `qtradex` (not a monorepo).
- **Public API** (from `qtradex.__init__`): `Data`, `load_csv`, `BaseBot`, `backtest`, `dispatch`, `live`, `papertrade`, `load_tune`, `PaperWallet`, `Wallet`, `Buy`, `Sell`, `Thresholds`, `Hold`, `ti` (tulipy wrapper), `plot`, `plotmotion`, `derivative`, `fitness`, `float_period`, `lag`, `expand_bools`, `rotate`, `truncate`.
- **CLI entry**: `qtradex-tune-manager` → `qtradex.core.tune_manager:main`
- **Bot base**: `qtradex.core.BaseBot` — override `indicators()`, `strategy()`, `fitness()`, `plot()`, `reset()`, `execution()`. `autorange()` auto-computes warmup days from tune keys ending in `_period`.
- **Tune params**: Keys ending in `_period` are auto-scaled to candle size (treated as day-count periods).
- **Clamps format**: `param → [min, midpoint, max, clamp_flag]` (`clamp_flag=0` to skip).
- **Dispatch**: `qx.dispatch(bot, data, wallet)` — interactive CLI (backtest/optimize/papertrade/live/AutoBacktest/MonteCarlo/ShowFillOrders).
- **Data**: `qtradex.public.Data(exchange, asset, currency, begin, end)` — fetches candles via CCXT, yfinance, etc.
- **Optimizers**: QPSO, LSGA, IPSE, AION, MouseWheelTuner in `qtradex.optimizers`.
- **IPC/candle cache**: `qtradex/common/pipe/` (gitignored) — JSON file-based concurrent IPC.
- **Tune caching**: per-bot `tunes/` next to the bot source (`demos/tunes/` gitignored).

## Platform
Linux-only (classifier: `Operating System :: POSIX :: Linux`).

## Signals
`Buy(price, maxvolume)`, `Sell(price, maxvolume)`, `Thresholds(buying, selling)`, `Hold`.
`is_override = True` → immediate execution (market order). `Thresholds` → limit orders (fill only when crossed).
