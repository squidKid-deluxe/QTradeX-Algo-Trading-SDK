"""
╔═╗╔╦╗╦═╗╔═╗╔╦╗╔═╗═╗ ╦
║═╬╗║ ╠╦╝╠═╣ ║║║╣ ╔╩╦╝
╚═╝╚╩ ╩╚═╩ ╩═╩╝╚═╝╩ ╚═

extinction_event.py

Extinction Event

Indicators;

    - ma1, ma2, ma3: Exponential Moving Averages (EMAs) with different periods
    - support: A support level based on a weighted combination of `ma1` and `ma2`
    - selloff: A selloff threshold based on a weighted combination of `ma1` and `ma2`
    - despair: A despair level based on a weighted combination of `ma1` and `ma2`
    - resistance: A resistance level based on a weighted combination of `ma1` and `ma2`
    - trend: The current market trend, which can be 'bull', 'bear', or None
    - buying: A calculated buying price based on the current trend and indicators
    - selling: A calculated selling price based on the current trend and indicators
    - override: Overrides the default behavior with 'buy' or 'sell' signals when a trend shift occurs


Strategy:

    Trending:

    - when the low price is first above the long average a qx.Buy is initiated indicating end of the Bear market
    - when the high price is first below the long average a qx.Sell is initiated indicating end of the Bull market

    Channeled:

    - During Bull Market a trading channel is defined by support and selloff
    - During Bear Market a trading channel is defined by despair and resistance
"""

import time

import numpy as np
import qtradex as qx


class ExtinctionEvent(qx.BaseBot):
    def __init__(self):
        self.tune = {
            "ma1_period": 5.8,
            "ma2_period": 15.0,
            "ma3_period": 30.0,
            "selloff ma1": 1.1,
            "selloff ma2": 1.1,
            "selloff ratio": 0.5,
            "support ma1": 1.0,
            "support ma2": 1.0,
            "support ratio": 0.5,
            "resistance ma1": 1.0,
            "resistance ma2": 1.0,
            "resistance ratio": 0.5,
            "despair ma1": 0.9,
            "despair ma2": 0.9,
            "despair ratio": 0.5,
        }

        self.clamps = {
            "ma1_period": [5, 5.8, 100, 1],
            "ma2_period": [5, 15.0, 100, 1],
            "ma3_period": [5, 30.0, 100, 1],
            "selloff ma1": [0.9, 1.1, 1.2, 1],
            "selloff ma2": [0.9, 1.1, 1.2, 1],
            "selloff ratio": [0.25, 0.5, 0.75, 1],
            "support ma1": [0.9, 1.0, 1.2, 1],
            "support ma2": [0.9, 1.0, 1.2, 1],
            "support ratio": [0.25, 0.5, 0.75, 1],
            "resistance ma1": [0.9, 1.0, 1.2, 1],
            "resistance ma2": [0.9, 1.0, 1.2, 1],
            "resistance ratio": [0.25, 0.5, 0.75, 1],
            "despair ma1": [0.9, 0.9, 1.2, 1],
            "despair ma2": [0.9, 0.9, 1.2, 1],
            "despair ratio": [0.25, 0.5, 0.75, 1],
        }

    def indicators(self, data):
        metrics = {
            tag.rsplit("_", 1)[0]: qx.ti.ema(data["close"], self.tune[tag])
            for tag in ["ma1_period", "ma2_period", "ma3_period"]
        }
        metrics["ma_exec"] = qx.ti.ema(data["close"], 2)
        metrics["support"] = []
        metrics["selloff"] = []
        metrics["despair"] = []
        metrics["resistance"] = []
        metrics["trend"] = []
        metrics["buying"] = []
        metrics["selling"] = []
        metrics["override"] = []

        trend = None

        for ma1, ma2, ma3, low, high in zip(
            metrics["ma1"], metrics["ma2"], metrics["ma3"], data["low"], data["high"]
        ):
            support = ma1 * self.tune["support ma1"] * self.tune[
                "support ratio"
            ] + ma2 * self.tune["support ma2"] * (1 - self.tune["support ratio"])
            selloff = ma1 * self.tune["selloff ma1"] * self.tune[
                "selloff ratio"
            ] + ma2 * self.tune["selloff ma2"] * (1 - self.tune["selloff ratio"])
            despair = ma1 * self.tune["despair ma1"] * self.tune[
                "despair ratio"
            ] + ma2 * self.tune["despair ma2"] * (1 - self.tune["despair ratio"])
            resistance = ma1 * self.tune["resistance ma1"] * self.tune[
                "resistance ratio"
            ] + ma2 * self.tune["resistance ma2"] * (1 - self.tune["resistance ratio"])

            # SECURITY: prevent flash trading risk
            support, selloff = sorted([support, selloff])
            despair, resistance = sorted([despair, resistance])

            metrics["selloff"].append(selloff)
            metrics["support"].append(support)
            metrics["resistance"].append(resistance)
            metrics["despair"].append(despair)

            if low > ma3 and trend != "bull":
                trend = "bull"
                metrics["override"].append("buy")
            elif high < ma3 and trend != "bear":
                trend = "bear"
                metrics["override"].append("sell")
            else:
                metrics["override"].append(None)

            if trend is None:
                metrics["buying"].append(metrics["ma3"][-1] / 2)
                metrics["selling"].append(metrics["ma3"][-1] * 2)
            elif trend == "bull":
                metrics["buying"].append(metrics["support"][-1])
                metrics["selling"].append(metrics["selloff"][-1])
            elif trend == "bear":
                metrics["buying"].append(metrics["despair"][-1])
                metrics["selling"].append(metrics["resistance"][-1])
            else:
                raise RuntimeError

            metrics["trend"].append(trend)

        return metrics

    def plot(self, data, states, indicators, block):
        axes = qx.plot(
            self.info,
            data,
            states,
            indicators,
            False,
            (
                # key, name, color, index, title
                ("ma3", "LONG", "white", 0, "Extinction Event"),
            ),
        )

        axes[0].fill_between(
            states["dates"],
            indicators["selloff"],
            indicators["support"],
            color="lime",
            alpha=0.3,
            where=[i == "bull" for i in indicators["trend"]],
            label="Support/Selloff",
            step="post",
        )

        axes[0].fill_between(
            states["dates"],
            indicators["resistance"],
            indicators["despair"],
            color="tomato",
            alpha=0.4,
            where=qx.expand_bools([i == "bear" for i in indicators["trend"]]),
            label="Resistance/Despair",
            step="post",
        )

        axes[0].legend()
        qx.plotmotion(block)

    def strategy(self, tick_info, indicators):
        if indicators["override"] == "buy" and isinstance(
            tick_info["last_trade"], (qx.Sell, qx.Thresholds)
        ):
            ret = qx.Buy()
        elif indicators["override"] == "sell" and isinstance(
            tick_info["last_trade"], (qx.Buy, qx.Thresholds)
        ):
            ret = qx.Sell()
        else:
            ret = qx.Thresholds(
                buying=indicators["buying"], selling=indicators["selling"]
            )
        return ret

    def execution(self, signal, indicators, wallet):
        if isinstance(signal, qx.Buy):
            ret = qx.Buy(indicators["ma_exec"])
        elif isinstance(signal, qx.Sell):
            ret = qx.Sell(indicators["ma_exec"])
        else:
            ret = signal
        return ret

    def fitness(self, states, raw_states, asset, currency):
        return [
            "roi_assets",
            "roi_currency",
            "roi",
            "cagr",
            "sortino",
            "maximum_drawdown",
            "trade_win_rate",
        ], {}


def main():
    asset, currency = "BTC", "USDT"
    wallet = qx.PaperWallet({asset: 0, currency: 1})
    data = qx.Data(
        exchange="kucoin",
        asset=asset,
        currency=currency,
        begin="2020-01-01",
        end="2025-01-01",
    )

    bot = ExtinctionEvent()
    qx.dispatch(bot, data, wallet)


if __name__ == "__main__":
    main()
