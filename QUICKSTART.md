# QTradeX: Writing a New Bot

**QTradeX** is a framework for creating and backtesting algorithmic trading bots. This guide walks through creating a bot from scratch.

For example bots, see [the QTradeX AI Agents repo](https://github.com/squidKid-deluxe/qtradex-ai-agents).

## Prerequisites

- **Python 3.9+**
- Install QTradeX:
  ```bash
  pip install qtradex
  ```
- Trading concepts (moving averages, etc.) helpful but not required.

---

## Creating a New Bot

Subclass `qx.BaseBot` and override these methods:

| Method | Purpose |
|---|---|
| `__init__` | Set `self.tune` (parameters) and `self.clamps` (bounds) |
| `indicators(data)` | Compute technical indicators from candle data |
| `strategy(tick_info, indicators)` | Return a signal: `Buy`, `Sell`, `Thresholds`, or `Hold` |
| `fitness(states, raw_states, asset, currency)` | Return `(metric_names, {})` — defaults `["roi", "cagr", "trade_win_rate"]` |
| `plot(data, states, indicators, block)` | Add indicator subplots (see example below) |
| `reset()` | Clear internal state between optimization runs (optional) |
| `execution(signal, indicators, wallet)` | Modify signals just before execution (optional) |

Tune keys ending in `_period` are auto-scaled to the candle size (treated as day-count periods).

### Example bot template

```python
import qtradex as qx

class Bot(qx.BaseBot):
    def __init__(self):
        self.tune = {
            "ema_period": 14.0,
            "buy_factor": 1.05,
            "sell_factor": 0.95,
        }
        self.clamps = {
            "ema_period": [5, 14, 100, 1],
            "buy_factor": [1.0, 1.05, 4.0, 1],
            "sell_factor": [1.0, 0.95, 4.0, 1],
        }

    def indicators(self, data):
        return {
            "ema": qx.ti.ema(data["close"], self.tune["ema_period"]),
        }

    def strategy(self, tick_info, indicators):
        price = tick_info["close"]
        ema = indicators["ema"]
        if price < ema * 0.98:
            return qx.Buy()
        elif price > ema * 1.02:
            return qx.Sell()
        return qx.Thresholds(buying=price * 0.99, selling=price * 1.01)

    def fitness(self, states, raw_states, asset, currency):
        return ["roi", "sortino", "maximum_drawdown"], {}
```

### Configuring bot parameters

`self.tune` holds parameter values — either set manually or tuned by an optimizer. `self.clamps` defines the search space:

```
param_name → [min, midpoint, max, clamp_flag]
```

Set `clamp_flag` to `0` to exclude a parameter from optimization, or omit the entry.

### Defining indicators

All Tulip indicator functions are available through `qx.ti` and are LRU-cached (256 entries):

```python
def indicators(self, data):
    return {
        "ema": qx.ti.ema(data["close"], self.tune["ema_period"]),
        "rsi": qx.ti.rsi(data["close"], 14),
    }
```

### Building a strategy

Return one of four signal types:

| Signal | Behavior |
|---|---|
| `qx.Buy(price=None, maxvolume=inf)` | Market buy — executes immediately |
| `qx.Sell(price=None, maxvolume=inf)` | Market sell — executes immediately |
| `qx.Thresholds(buying, selling)` | Place limit orders at threshold prices |
| `qx.Hold()` | Cancel all orders / do nothing |

`Buy`/`Sell` have `is_override = True` (immediate execution). `Thresholds` has `is_override = False` (limit orders fill only when crossed).

---

## Testing and Backtesting

Use `qx.dispatch(bot, data, wallet)` for an interactive CLI menu (backtest / optimize / papertrade / live):

```python
def main():
    asset, currency = "BTC", "USDT"
    wallet = qx.PaperWallet({asset: 1, currency: 0})
    data = qx.Data(
        exchange="kucoin",
        asset=asset,
        currency=currency,
        begin="2021-01-01",
        end="2023-01-01",
    )
    bot = Bot()
    qx.dispatch(bot, data, wallet)
```

You can also call `qx.backtest(bot, data, wallet)` directly to skip the interactive menu and get results as a dict.

---

## Plotting and Visualization

Override `plot()` in your bot class to add indicator subplots:

```python
def plot(self, data, states, indicators, block):
    qx.plot(
        self.info, data, states, indicators, block,
        (
            ("ema", "EMA", "cyan", 0, "Exponential Moving Average"),
        ),
    )
    qx.plotmotion(block)
```

The `(key, label, color, axis_index, axis_title)` tuples define indicator overlays. Axis 0 is the main price chart.

---

## Optimizing the Bot

Select "Optimize" from the `qx.dispatch()` menu, or use an optimizer directly:

```python
from qtradex.optimizers import QPSO

optimizer = QPSO(data, wallet)
optimizer.optimize(bot)
```

Available optimizers in `qtradex.optimizers`: `QPSO`, `LSGA`, `IPSE`, `AION`, `MouseWheelTuner`.

---

## Deployment

After testing and optimization, deploy live by selecting "Live" or "Papertrade" from the `qx.dispatch()` menu. You will be prompted for exchange API credentials.

---

## Resources

- [QTradeX GitHub](https://github.com/squidKid-deluxe/QTradeX-Algo-Trading-SDK)
- [QTradeX AI Agents (example bots)](https://github.com/squidKid-deluxe/qtradex-ai-agents)
- [Tulipy docs](https://tulipindicators.org)
- [CCXT docs](https://docs.ccxt.com)

