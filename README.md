# 🚀 QTradeX Core — Build, Backtest & Optimize AI-Powered Crypto Trading Bots

<p>
  <img src="https://img.shields.io/github/stars/squidKid-deluxe/QTradeX-Algo-Trading-SDK" />
  <img src="https://img.shields.io/github/contributors/squidKid-deluxe/QTradeX-Algo-Trading-SDK" />
  <img src="https://img.shields.io/github/last-commit/squidKid-deluxe/QTradeX-Algo-Trading-SDK" />
  <img src="https://visitor-badge.laobi.icu/badge?page_id=squidKid-deluxe.QTradeX-Algo-Trading-SDK" />
  <img src="https://img.shields.io/github/languages/count/squidKid-deluxe/QTradeX-Algo-Trading-SDK" />
  <img src="https://img.shields.io/github/languages/top/squidKid-deluxe/QTradeX-Algo-Trading-SDK" />
  <img src="https://img.shields.io/github/issues/squidKid-deluxe/QTradeX-Algo-Trading-SDK" />
  <img src="https://img.shields.io/github/issues-pr/squidKid-deluxe/QTradeX-Algo-Trading-SDK" />
</p>

<p align="center">
  <img src="screenshots/Screenshot from 2025-05-02 18-50-54.png" width="100%" alt="QTradeX Demo Screenshot">
</p>

> 📸 See [screenshots.md](screenshots.md) for more visuals  
> 📚 Read the core docs on [QTradeX SDK DeepWiki](https://deepwiki.com/squidKid-deluxe/QTradeX-Algo-Trading-SDK)  
> 🤖 Explore the bots at [QTradeX AI Agents DeepWiki](https://deepwiki.com/squidKid-deluxe/QTradeX-AI-Agents)  
> 💬 Join our [Telegram Group](https://t.me/qtradex_sdk) for discussion & support

---

## TL;DR
**QTradeX** is a lightning-fast Python framework for designing, backtesting, and deploying algorithmic trading bots, built for **crypto markets** with support for **100+ exchanges**, **AI-driven optimization**, and **blazing-fast vectorized execution**.

Like what we're doing?  Give us a ⭐!

---

## Why QTradeX?

Whether you're exploring a simple EMA crossover or engineering a strategy with 20+ indicators and genetic optimization, QTradeX gives you:

- Modular, non-locked architecture - want to use QTradeX's data fetching with a custom backtest engine?  Go for it!
- Tulip + CCXT Integration  
- Custom Bot Classes  
- Fast, Disk-Cached Market Data  
- Ultra Fast Backtests (even on a Raspberry Pi!)

---

## 🔍 Features at a Glance

- **Bot Development**: Extend `BaseBot` to craft custom strategies
- **Backtesting**: Easy-to-navigate CLI & live-coding based testing platform (Just select `Autobacktest`)
- **Optimization**: Use QPSO, LSGA, or others to fine-tune parameters
- **Indicators**: Wrapped Tulip indicators for blazing performance
- **Data Sources**: Pull candles from 100+ CEXs/DEXs with CCXT
- **Performance Metrics**: Evaluate bots with ROI, Sortino, Win Rate, and dozens more
- **Speed**: 200+ backtests per second for 3 years of daily candles on a Ryzen 5600x

---

## Project Structure

```
qtradex/
├── core/             # Bot logic and backtesting
├── indicators/       # Technical indicators
├── optimizers/       # QPSO, LSGA, other optimizers, and common utilities
├── plot/             # Trade/metric visualization
├── private/          # Execution & paper wallets
├── public/           # Data feeds and utils
└── common/           # JSON RPC, BitShares nodes, and data caching
```

---

## Quickstart

### Install

```bash
pip install qtradex
```

Or, if you want the latest updates:

```bash
git clone https://github.com/squidKid-deluxe/QTradeX-Algo-Trading-SDK.git QTradeX
cd QTradeX
pip install -e .
````

---

## Example Bot: EMA Crossover

```python
import qtradex as qx
import numpy as np


class EMACrossBot(qx.BaseBot):
    def __init__(self):
        # Notes:
        # - If you make the tune values integers, the optimizers
        #   will quantize them to the nearest integer.
        # - By putting `_period` at the end of a tune value,
        #   QTradeX core will assume they are periods in days and will scale them
        #   to different candle sizes if the data given isn't daily
        self.tune = {
            "fast_ema_period": 10.0,
            "slow_ema_period": 50.0
        }
        self.clamps = [
            # min, max
            [5,   50 ], # fast_ema
            [20,  100], # slow_ema
        ]

    def indicators(self, data):
        return {
            "fast_ema": qx.ti.ema(data["close"], self.tune["fast_ema"]),
            "slow_ema": qx.ti.ema(data["close"], self.tune["slow_ema"]),
        }

    def strategy(self, tick_info, indicators):
        fast = indicators["fast_ema"]
        slow = indicators["slow_ema"]
        if fast > slow:
            return qx.Buy()
        elif fast < slow:
            return qx.Sell()
        return qx.Thresholds(buying=fast * 0.8, selling=fast * 1.2)

    def plot(self, *args):
        qx.plot(
            self.info,
            *args,
            (
                # key name    label    color   axis idx   axis name
                ("fast_ema", "EMA 1", "white", 0,        "EMA Cross"),
                ("slow_ema", "EMA 2", "cyan",  0,        "EMA Cross"),
            )
        )


# Load data and run
data = qx.Data(
    exchange="kucoin",
    asset="BTC",
    currency="USDT",
    begin="2020-01-01",
    end="2023-01-01"
)
bot = EMACrossBot()
qx.dispatch(bot, data)
```

See more bots in [QTradeX AI Agents](https://github.com/squidKid-deluxe/QTradeX-AI-Agents)

---

## Usage Guide

| Step | What to Do                                                      |
| ---- | --------------------------------------------------------------- |
| 1️⃣  | Build a bot with custom logic by subclassing `BaseBot`          |
| 2️⃣  | Backtest using `qx.core.dispatch` + historical data             |
| 3️⃣  | Optimize with any algorithm you like (optmized tunes stored in `/tunes`) |
| 4️⃣  | Deploy live                                                     |

---

## Roadmap

* More indicators (non-Tulip sources)
* GPU Acceleration for indicators
* Improved multi-core support for optimization
* Windows/Mac support
* TradFi Connectors: Stocks, Forex, and Comex support

Want to help out?  Check out the [Issues](https://github.com/squidKid-deluxe/QTradeX-Algo-Trading-SDK/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22) list for forseeable improvements and bugs.

---

## Resources

* [QTradeX Telegram](https://t.me/qtradex_sdk)
* [QTradeX Algo Trading Strategies](https://github.com/squidKid-deluxe/qtradex-ai-agents)
* [Tulipy Docs](https://tulipindicators.org)
* [CCXT Docs](https://docs.ccxt.com)

---

## 📜 License

**WTFPL** — Do what you want. Just be awesome about it 😎

---

## ⭐ Star History

<a href="https://www.star-history.com/#squidKid-deluxe/QTradeX-Algo-Trading-SDK&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=squidKid-deluxe/QTradeX-Algo-Trading-SDK&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=squidKid-deluxe/QTradeX-Algo-Trading-SDK&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=squidKid-deluxe/QTradeX-Algo-Trading-SDK&type=Date" />
 </picture>
</a>

---

✨ Ready to start? Clone the repo, run your first bot, and tune away.  Once tuned - LET THE EXECUTIONS BEGIN!
