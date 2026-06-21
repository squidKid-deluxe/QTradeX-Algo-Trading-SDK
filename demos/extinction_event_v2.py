"""
Extinction Event — v2 API port (single-asset, uses Allocation/Limit).
Mirrors demos/extinction_event.py for result comparison once v2 engine is built.
"""

import numpy as np
import qtradex as qx
from qtradex.private.signals import Buy, Sell


TOKEN = "BTC"


class ExtinctionEventV2(qx.BaseBot):
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
        c = data[TOKEN]["close"]
        metrics = {
            tag.rsplit("_", 1)[0]: qx.ti.ema(c, self.tune[tag])
            for tag in ["ma1_period", "ma2_period", "ma3_period"]
        }
        metrics["ma_exec"] = qx.ti.ema(c, 2)
        metrics["support"] = []
        metrics["selloff"] = []
        metrics["despair"] = []
        metrics["resistance"] = []
        metrics["trend"] = []
        metrics["buying"] = []
        metrics["selling"] = []
        metrics["override"] = []

        trend = None
        low = data[TOKEN]["low"]
        high = data[TOKEN]["high"]

        for ma1, ma2, ma3, l, h in zip(
            metrics["ma1"], metrics["ma2"], metrics["ma3"], low, high
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

            support, selloff = sorted([support, selloff])
            despair, resistance = sorted([despair, resistance])

            metrics["selloff"].append(selloff)
            metrics["support"].append(support)
            metrics["resistance"].append(resistance)
            metrics["despair"].append(despair)

            if l > ma3 and trend != "bull":
                trend = "bull"
                metrics["override"].append("buy")
            elif h < ma3 and trend != "bear":
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

    def strategy(self, tick_info, indicators):
        last = tick_info.get("last_trade")
        if indicators["override"] == "buy" and isinstance(last, Sell):
            return qx.Allocation(targets={TOKEN: 1.0})
        elif indicators["override"] == "sell" and isinstance(last, Buy):
            return qx.Allocation(targets={TOKEN: 0.0})
        else:
            return qx.Allocation(
                limits={
                    TOKEN: qx.Limit(
                        {indicators["buying"]: 1.0, indicators["selling"]: 0.0}
                    )
                }
            )

    def execution(self, allocation, indicators, wallet, prices):
        signals = super().execution(allocation, indicators, wallet, prices)
        for s in signals:
            if isinstance(s, (Buy, Sell)):
                s.price = indicators["ma_exec"]
        return signals

    def fitness(self, states, raw_states):
        return [
            "roi",
            "cagr",
            "sortino",
            "maximum_drawdown",
            "trade_win_rate",
        ], {}

    def plot(self, data, states, indicators, block):
        axes = qx.plot(
            self.info,
            data,
            states,
            indicators,
            False,
            (
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


def main():
    wallet = qx.PaperWallet({TOKEN: 0, "USDT": 1})
    data = qx.Data(
        exchange="kucoin",
        assets=[TOKEN],
        currency="USDT",
        begin="2020-01-01",
        end="2025-01-01",
    )
    bot = ExtinctionEventV2()
    qx.dispatch(bot, data, wallet)


if __name__ == "__main__":
    main()
