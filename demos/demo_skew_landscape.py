"""Demo: 1D noise landscape evolving under LSGA with skew memory."""

import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import qtradex as qx

SEED = 42
KNOTS = 30

_rng = np.random.RandomState(SEED)
_knot_x = np.linspace(0, 1, KNOTS)
_knot_y = _rng.uniform(-1, 1, KNOTS)


def _noise_landscape(xs):
    return np.interp(xs, _knot_x, _knot_y)


def _fake_backtest(
    bot, data, wallet,
    plot=True, block=True, return_states=False,
    range_periods=True, show=True, fine_data=None, always_trade="smart",
):
    x = float(bot.tune.get("x", 0.5))
    keys, custom = bot.fitness(None, None, None, None)
    ret = {k: custom[k] for k in keys}
    if return_states:
        n = max(2, int(data.days * 86400 / data.candle_size))
        unix = np.linspace(data.begin + data.candle_size, data.end, n)
        close = np.full(n, 100.0)
        growth = _noise_landscape(np.array([x]))[0] * 0.5
        balances = [
            {data.asset: 1.0 + growth * i / (n - 1), data.currency: 0.0}
            for i in range(n)
        ]
        raw_states = {"unix": unix, "close": close, "balances": balances, "trades": []}
        states = {
            "detailed_trades": [],
            "begin": data.begin,
            "end": data.end,
            "candle_size": data.candle_size,
        }
        return ret, raw_states, states
    return ret


qx.backtest = _fake_backtest
qx.core.backtest = _fake_backtest

import qtradex.optimizers.lsga as lsga_mod
lsga_mod.backtest = _fake_backtest
from qtradex.core.base_bot import BaseBot
from qtradex.private.wallet import PaperWallet


class NoiseBot(BaseBot):
    def __init__(self):
        self.tune = {"x": np.float64(0.5)}
        self.clamps = {"x": [0.0, 0.5, 1.0, 1]}

    def indicators(self, data):
        return {}

    def strategy(self, state, indicators):
        pass

    def fitness(self, states, raw_states, asset, currency):
        x = float(self.tune.get("x", 0.5))
        return ["roi"], {"roi": float(_noise_landscape(np.array([x]))[0])}

    def reset(self):
        pass

    def execution(self, signal, indicators, wallet):
        return signal

    def autorange(self):
        return 1


class MockData:
    asset = "TEST"
    currency = "USD"
    exchange = "mock"
    begin = 0
    end = 86400
    candle_size = 60
    fine_data = None

    @property
    def days(self):
        return (self.end - self.begin) / 86400


# ---- Landscape capture -----

snapshots = []  # list of (xs, raw, eff, memory_copy)


def capture(sigma):
    xs = np.linspace(0, 1, 500)
    raw = _noise_landscape(xs)
    eff = raw.astype(float).copy()
    for i, x in enumerate(xs):
        cand = np.array([x])
        for stored_norm, sp in lsga_mod._skew_memory:
            d = np.linalg.norm(cand - stored_norm)
            w = np.exp(-0.5 * (d / sigma) ** 2)
            eff[i] *= 1.0 - (1.0 - sp) * w
    snapshots.append((xs.copy(), raw.copy(), eff.copy(), list(lsga_mod._skew_memory)))


class _CapturingList(list):
    def __init__(self, sigma):
        super().__init__()
        self._sigma = sigma

    def append(self, item):
        super().append(item)
        capture(self._sigma)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        capture(self._sigma)


if __name__ == "__main__":
    sigma = 0.01
    lsga_mod._skew_memory = _CapturingList(sigma)

    capture(sigma)

    bot = NoiseBot()
    opt = lsga_mod.LSGAoptions()
    opt.population = 15
    opt.offspring = 5
    opt.epochs = 100
    opt.skew_check_period = 10
    opt.skew_mc_iterations = 5
    opt.improvements = 100000
    opt.show_terminal = False

    data = MockData()
    wallet = PaperWallet({"TEST": 0, "USD": 1})
    optimizer = lsga_mod.LSGA(data, wallet, opt)
    optimizer.optimize(bot)

    print(f"Captured {len(snapshots)} frames ({len(snapshots)-1} MC checks)")

    fig, ax = plt.subplots(figsize=(10, 6))

    def animate(i):
        ax.clear()
        xs, raw, eff, mem = snapshots[i]
        ax.plot(xs, raw, "b-", alpha=0.35, label="raw landscape", linewidth=1.5)
        ax.plot(xs, eff, "r-", label="effective landscape", linewidth=2)
        for stored_norm, _ in mem:
            ax.axvline(stored_norm[0], color="gray", alpha=0.25, linestyle="--")
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(0, 1)
        ax.set_title(f"Check {i}  —  blacklisted tunes: {len(mem)}", fontsize=13)
        ax.set_xlabel("x")
        ax.set_ylabel("fitness")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.15)

    ani = animation.FuncAnimation(
        fig, animate, frames=len(snapshots), interval=800, repeat=True
    )

    gif_path = os.path.join(os.path.dirname(__file__), "skew_landscape_demo.gif")
    try:
        ani.save(gif_path, writer="pillow", fps=1.25)
        print(f"Saved {gif_path}")
    except Exception as e:
        print(f"GIF save failed ({e}), showing interactively")
        plt.show()
