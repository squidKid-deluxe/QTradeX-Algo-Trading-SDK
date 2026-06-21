import json
import math
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import qtradex as qx
from qtradex.common.utilities import NdarrayEncoder

mplstyle.use("dark_background")
plt.rcParams["figure.raise_window"] = False


def bound_neurons(bot):
    def clamp(value, minv, maxv, strength):
        """
        clamp `value` between `minv` and `maxv` with `strength`
        if strength is one, value is hard clipped
        if strength is 0.5, value is returned as the mean of itself and any boundries it is outside of
        if strength is 0, it is returned as is

        this works for all values of strength between 0 and 1
        """

        isint = isinstance(value, int)

        ret = None
        # if told not to tune or value is within range
        if not strength or minv <= value <= maxv:
            # don't touch
            ret = value
        # less than minimum
        elif value < minv:
            ret = (value * (1 - strength)) + (minv * strength)
        # more than maximum
        elif value > maxv:
            ret = (value * (1 - strength)) + (maxv * strength)
        return int(ret) if isint else ret

    def ndclamp(value, minv, maxv, strength):
        """
        Clamp `value` between `minv` and `maxv` with `strength`.
        If strength is one, value is hard clipped.
        If strength is 0.5, value is returned as the mean of itself and any boundaries it is outside of.
        If strength is 0, it is returned as is.

        This works for all values of strength between 0 and 1.
        """
        
        # Create a mask for values less than minv
        less_than_min = value < minv
        # Create a mask for values greater than maxv
        greater_than_max = value > maxv
        
        # Initialize the result with the original value
        ret = np.copy(value)
        
        # Apply clamping for values less than minv
        if np.any(less_than_min):
            ret[less_than_min] = (value[less_than_min] * (1 - strength)) + (minv * strength)
        
        # Apply clamping for values greater than maxv
        if np.any(greater_than_max):
            ret[greater_than_max] = (value[greater_than_max] * (1 - strength)) + (maxv * strength)
        
        # Return as int if the original value was an integer
        if np.issubdtype(value.dtype, np.integer):
            return ret.astype(int)
        
        return ret

    bot.tune = {
        key: (
            ndclamp(bot.tune[key], minv, maxv, clamp_amt)
            if isinstance(bot.tune[key], np.ndarray)
            else clamp(bot.tune[key], minv, maxv, clamp_amt)
        )
        for key, (minv, _, maxv, clamp_amt) in bot.clamps.items()
    }

    bot.autorange()
    return bot


def print_tune(score, bot, render=False):
    msg = ""
    just = max(map(len, score))
    for k, s in score.items():
        msg += f"# {k}".ljust(just + 2) + f" {s:.3f}\n"

    msg += "self.tune = " + json.dumps(bot.tune, indent=4, cls=NdarrayEncoder)
    msg += "\n\n"
    if not render:
        print(msg)
    return msg


def end_optimization(best_bots, show):
    msg = "\033c=== FINAL TUNES ===\n\n"
    for key, value in best_bots.items():
        # ponytail: metadata keys (wf_intensity, etc.) are not (score, bot) tuples — skip them
        if not isinstance(value, tuple) or len(value) != 2:
            continue
        score, bot = value
        coord = key
        name = f"BEST {coord.upper()} TUNE"
        msg += "## " + name + "\n\n"
        msg += print_tune(score, bot, render=True)
        save_bot = deepcopy(bot)
        save_bot.tune = {"tune": bot.tune.copy(), "results": score}
        qx.core.tune_manager.save_tune(save_bot, name)
    if show:
        print(msg)


def merge(tune1, tune2):
    tune3 = {}
    for k, v in tune1.items():
        value = (random.random() / 2) + 0.25
        if isinstance(v, int):
            tune3[k] = int(round((v * value) + (tune2[k] * (1 - value))))
        else:
            tune3[k] = (v * value) + (tune2[k] * (1 - value))
    return tune3


def plot_scores(historical, historical_tests, cdx):
    """
    historical is a matrix like this:
    [
        (
            idx,
            [
                (score, bot),
                (score, bot),
                ...
            ]
        )
    ]
    """
    if not historical:
        return
    plt.clf()
    n_coords = len(historical[0][1])
    coords = list(historical[0][1].keys())
    # initialize empty lists
    lines = [[] for _ in range(n_coords)]
    x_list = []
    for mdx, moment in enumerate(historical):
        x_list.append(moment[0])
        if mdx:
            x_list.append(moment[0])
            for idx in range(n_coords):
                lines[idx].append(lines[idx][-1])

        for coord, (score, _) in moment[1].items():
            lines[coords.index(coord)].append(score[coord])
    x_list.append(cdx)
    for idx in range(n_coords):
        lines[idx].append(lines[idx][-1])

    sqrt = n_coords**0.5

    height = math.ceil(sqrt)
    width = height

    x_list_tests = [i[0] for i in historical_tests]

    for idx, coord in enumerate(coords):
        plt.subplot(width, height, idx + 1)
        plt.title(coord)
        plt.plot(x_list, lines[idx], color="green")
        plt.xscale("log")
        plt.scatter(
            x_list_tests,
            [i[1][coord] for i in historical_tests],
            color="yellow",
        )
    plt.tight_layout()
    plt.pause(0.1)
