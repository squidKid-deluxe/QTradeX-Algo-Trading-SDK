import json
import shutil
import time
from getpass import getpass
from random import choice, sample

import numpy as np
import qtradex as qx
from qtradex.common.utilities import it
from qtradex.core.tune_manager import choose_tune
from qtradex.core.tune_manager import load_tune as load_from_manager
from qtradex.core.ui_utilities import get_number, logo, select
from qtradex.private.wallet import PaperWallet


def load_tune(bot):
    options = [
        "Use best roi tune",
        "Use most recent best roi tune",
        "Use bot.tune",
        "Use bot.drop",
        "Use tune manager...",
    ]
    choice = select(options)
    if choice == 0:
        return load_from_manager(bot)
    elif choice == 1:
        return load_from_manager(bot, sort="latest")
    elif choice == 2:
        return bot.tune
    elif choice == 3:
        return {k: v[1] for k, v in bot.clamps.items()}
    elif choice == 4:
        return choose_tune(bot, "tune")

def plot_gravitas(bot, data, wallet, **kwargs):
    import matplotlib.pyplot as plt

    def get_float_input(prompt, default):
        user_input = input(f"{prompt} (default: {default}): ")
        return float(user_input) if user_input else default

    # Get three float inputs with default values
    min_g = get_float_input("Min Gravitas", 0.3)
    max_g = get_float_input("Max Gravitas", 1.7)
    tests = int(get_float_input("Number of tests", 200.0))
    qx.backtest(bot, data, wallet.copy(), **kwargs, block=False)
    plt.figure("Gravitas")
    rois = []
    for g in np.linspace(min_g, max_g, tests):
        bot.gravitas = g
        rois.append(qx.backtest(bot, data, wallet.copy(), plot=False, **kwargs)["roi"])

    plt.plot(np.linspace(min_g, max_g, tests), rois)
    plt.ioff()
    plt.show()


def dispatch(bot, data, wallet=None, **kwargs):
    if wallet is None:
        wallet = PaperWallet({data.asset: 0, data.currency: 1})
    logo(animate=True)

    bot.tune = load_tune(bot)
    options = [
        "Backtest",
        "Optimize",    
        "Papertrade",
        "Live",
        "Show Fill Orders",
        "AutoBacktest",
        "Monte Carlo",
    ]
    choice = select(options)

    if choice == 0:
        qx.core.backtest(bot, data, wallet, **kwargs)
    elif choice == 1:
        for k, v in bot.clamps.items():
            if len(v) == 2:
                bot.clamps[k] = (v[0], (v[0]+v[1]) / 2, v[1], 1)

        options = [
            "QPSO (Quantum Particle Swarm Optimizer)",
            "LSGA (Local Search Genetic Algorithm)",
            "IPSE (Iterative Parametric Space Expansion)",
            "AION (Adaptive Intelligent Optimization Network)",
            "Manual Tuner",
            "Gravitas",
        ]
        choice = select(options)

        if choice == 0:
            optimizer = qx.optimizers.QPSO(data, wallet)
        elif choice == 1:
            optimizer = qx.optimizers.LSGA(data, wallet)
        elif choice == 2:
            optimizer = qx.optimizers.IPSE(data, wallet)
        elif choice == 3:
            optimizer = qx.optimizers.AION(data, wallet)
        elif choice == 4:
            optimizer = qx.optimizers.MouseWheelTuner(data, wallet)
        elif choice == 5:
            plot_gravitas(bot, data, wallet, **kwargs)
        if choice != 4:
            optimizer.optimize(bot, **kwargs)
    elif choice == 2:
        qx.core.papertrade(bot, data, wallet, **kwargs)
    elif choice in [3, 4]:
        if data.exchange == "bitshares":
            api_key = input("Enter username: ")
            api_secret = getpass("Enter WIF:      ")
        else:
            api_key = getpass("Enter API key:    ")
            api_secret = getpass("Enter API secret: ")

        if choice == 3:
            dust = input("Don't trade under this amount of assets (enter for 1e-8): ")
            if dust == "":
                dust = 1e-8
            else:
                dust = float(dust)

        # TODO:
        # some kind of login menu, currently an error is thrown if the key isn't valid

        if choice == 3:
            qx.core.live(bot, data, api_key, api_secret, dust, **kwargs)
        elif choice == 4:
            qx.core.filltest(bot, data, api_key, api_secret)
    elif choice == 5:
        qx.core.auto_backtest(bot, data, wallet, **kwargs)
    elif choice == 6:
        qx.core.monte_carlo(bot, data, wallet, **kwargs)
