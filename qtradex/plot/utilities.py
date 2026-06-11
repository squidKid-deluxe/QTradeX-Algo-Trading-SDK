import os
import platform
import sys
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
from matplotlib.collections import LineCollection
from qtradex.common.utilities import NIL, expand_bools, rotate, sigfig
from qtradex.private.signals import Buy, Sell

if not os.environ.get("MPLBACKEND"):
    try:
        import tkinter
    except ImportError:
        os.environ.setdefault("MPLBACKEND", "Agg")
    matplotlib.use("TkAgg")

def maximize_window():
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())


def plotmotion(block):
    if block:
        plt.ioff()
        plt.show()
    else:
        plt.ion()
        plt.pause(0.00001)


def plot_indicators(axes, timestamps, states, indicators, indicator_fmt):
    textdxes = []
    if not indicators:
        return
    for key, name, color, idx, title in indicator_fmt:
        ax = axes[idx]
        if idx and idx not in textdxes:
            textdxes.append(idx)
            ax.text(
                0.5,
                0.15,
                title,
                transform=ax.transAxes,
                color="white",
                alpha=0.15,
                ha="center",
                va="center",
                zorder=-1,
                fontdict={"fontsize": 40, "fontweight": "bold", "family": "monospace"},
            )
        # Plot each EMA with a color gradient
        ax.step(timestamps, indicators[key], color=color, label=name, where="post")


def unix_to_stamp(unix):
    if isinstance(unix, (float, int)):
        return datetime.fromtimestamp(unix)
    else:
        return [datetime.fromtimestamp(i) for i in unix]


def plot(
    info,
    data,
    states,
    indicators,
    block,
    indicator_fmt,
    style="dark_background",
):
    """
    plotting of buy/sell with win/loss line plotting
    buy/sell are green/red triangles
    plotting of high/low/open/close
    plotting of indicators (dict of indicator keys to be plotted and color)
    balance plotting follows price on token not held

    During papertrade and live sessions, the plotting is a bit different.

    Notably:
    - the red and green `open - close` clouds are not displayed
    - an extra argument, `raw` is given as raw high-frequency data and the high/low for
      that is plotted instead
    - past live trades are passed in and no "backtest trades" are plotted past the earliest
    """
    mplstyle.use(style)

    n_levels = (max(i[3] for i in indicator_fmt) if indicator_fmt else 0) + 2
    # clear the current figure
    plt.clf()
    fig = plt.gcf()
    gs = matplotlib.gridspec.GridSpec(
        n_levels, 1, height_ratios=[2] + ([1] * (n_levels - 1))
    )
    axes = [fig.add_subplot(i) for i in gs]

    title = [i[4] for i in indicator_fmt if not i[3]]
    if title and title[-1] is not None:
        title = title[-1]
    else:
        title = (os.path.split(sys.argv[0])[1]
        .rsplit(".py", 1)[0]
        .replace("_", " ")
        .title())
    axes[0].text(
        0.5,
        0.9,
        title,
        transform=axes[0].transAxes,
        color="mediumorchid",
        alpha=0.4,
        ha="center",
        va="center",
        zorder=-1,
        fontdict={"fontsize": 40, "fontweight": "bold", "family": "monospace"},
    )
    axes[0].yaxis.tick_right()
    axes[0].yaxis.set_label_position("right")
    axes[0].set_yscale("log")
    axes[0].text(
        0.5, # center x
        0.5, # center y
        r"""     ____                                    __     __  
    / __ \    ____  ____   __   ____  ____  (_ \   / _) 
   / /  \ \  (_  _)(  _ \ / _\ (    \(  __)   \ \_/ /   
  | |    | |   ||   )   //    \ ) D ( ) _)     \   /    
  | |  /\| |  (__) (__\_)\_/\_/(____/(____)    / _ \    
   \ \_\ \/                                  _/ / \ \_  
    \___\ \_                                (__/   \__) 
         \__)                                           
""",
        transform=axes[0].transAxes,
        color="cyan",
        alpha=0.15,
        ha="center",
        va="center",
        zorder=-1,
        fontdict={"fontsize": 20, "fontweight": "bold", "family": "monospace"},
    )

    # Add a left side title of "MODE / PAIR / EXCHANGE"
    axes[0].text(
        -0.02, # out of bounds left x
        0.5, # center y
        f'{info["mode"].upper()}:  {data.asset}/{data.currency} AT {data.exchange.upper()}',
        transform=axes[0].transAxes,
        fontsize=13,
        verticalalignment="center",
        horizontalalignment="center",
        rotation=90,
        color="deepskyblue",
    )

    timestamps = unix_to_stamp(states["unix"])
    states["dates"] = timestamps

    # plotting of high/low/open/close
    # high/low
    axes[0].fill_between(
        timestamps,
        states["low"],
        states["high"],
        color="mediumorchid",
        alpha=0.15,
        label="High/Low",
        step="post",
    )
    if info["mode"] not in ["live", "papertrade"]:
        axes[0].step(
            timestamps, states["close"], color="white", where="post", linewidth=2
        )
        axes[0].step(
            timestamps,
            states["close"],
            color="deepskyblue",
            where="post",
            linewidth=0.6,
        )
    if "live_data" in info:
        high_res = info["live_data"]
        # mindx = np.searchsorted(high_res["unix"], states["unix"][0], side="left")
        # high_res = {k: v[-mindx:] for k, v in high_res.items()}

        # Fill between for open > close
        axes[0].fill_between(
            unix_to_stamp(high_res["unix"]),
            high_res["high"],
            high_res["low"],
            color=(1, 1, 1, 0.8),  # white green
            label="high > low",
            step="post",
        )

        # Fill between for open > close
        axes[0].fill_between(
            unix_to_stamp(high_res["unix"]),
            high_res["open"],
            high_res["close"],
            where=expand_bools(high_res["open"] > high_res["close"], side="right"),
            color=(1, 0.8, 0.8, 1),  # white red
            label="open > close",
            step="post",
        )

        # Fill between for open < close
        axes[0].fill_between(
            unix_to_stamp(high_res["unix"]),
            high_res["open"],
            high_res["close"],
            where=expand_bools(high_res["open"] < high_res["close"], side="right"),
            color=(0.8, 0.8, 1, 1),  # white blue
            label="open < close",
            step="post",
        )
        # axes[0].vlines(
        #     unix_to_stamp(info["start"]),
        #     min(high_res["low"]),
        #     max(high_res["high"]),
        #     color="yellow",
        #     linestyles="dashed",
        # )

        axes[0].axvline(unix_to_stamp(info["start"]), color='yellow', alpha=0.5, linestyle='--')

    # Plot Live Fill Orders
    if "live_trades" in info:
        print(info["live_trades"])


        # FIXME need to add keys [timestamp, price, side] to bitshares info["live_trades"] for fill order plotting
        # This should happen when creating info["live_trades"], not here


        # [{'id': 'TWGV6U-WBJJE-PXU4NS', 'order': 'OGSITQ-AYAZ4-XTNMVE', 'info': {'ordertxid': 'OGSITQ-AYAZ4-XTNMVE', 'postxid': 'TKH2SE-M7IF5-CFI7LT', 
        # 'pair': 'XXBTZUSD', 'aclass': 'forex', 'time': '1744219378.616529', 'type': 'sell', 'ordertype': 'limit', 
        # 'price': '79953.00000', 'cost': '158.03510', 'fee': '0.31607', 'vol': '0.00197660', 'margin': '0.00000', 
        # 'leverage': '0', 'misc': '', 'trade_id': '81933107', 'maker': True, 'id': 'TWGV6U-WBJJE-PXU4NS'}, 
        # 'timestamp': 1744219378616, 'datetime': '2025-04-09T17:22:58.616Z', 'symbol': 'BTC/USD', 
        # 'type': 'limit', 'side': 'sell', 'takerOrMaker': 'maker', 
        # 'price': 79953.0, 'amount': 0.0019766, 'cost': 158.0351, 'fee': {'currency': 'USD', 'cost': 0.31607}, 
        # 'fees': [{'currency': 'USD', 'cost': 0.31607}]}, {'id': 'TM7GAS-WTVVQ-2Z62Y4', 'order': 'O5NRLR-IF2LQ-B6FLXS', 
        # 'info': {'ordertxid': 'O5NRLR-IF2LQ-B6FLXS', 'postxid': 'TKH2SE-M7IF5-CFI7LT', 'pair': 'XXBTZUSD', 'aclass': 'forex', 
        # 'time': '1744300278.272592', 'type': 'buy', 'ordertype': 'limit', 'price': '78617.80000', 'cost': '157.69945', 
        # 'fee': '0.31540', 'vol': '0.00200590', 'margin': '0.00000', 'leverage': '0', 'misc': '', 
        # 'trade_id': '81991541', 'maker': True, 'id': 'TM7GAS-WTVVQ-2Z62Y4'}, 'timestamp': 1744300278272, 
        # 'datetime': '2025-04-10T15:51:18.272Z', 'symbol': 'BTC/USD', 'type': 'limit', 'side': 'buy', 
        # 'takerOrMaker': 'maker', 'price': 78617.8, 'amount': 0.0020059, 'cost': 157.69945, 
        # 'fee': {'currency': 'USD', 'cost': 0.3154}, 'fees': [{'currency': 'USD', 'cost': 0.3154}]}, ...]



        # {'id': '5.0.198843620', 'key': {'base': '1.3.4157', 'quote': '1.3.5589', 'sequence': -691087}, 
        # 'time': '2025-04-20T20:42:30', 'op': {'fee': {'amount': 1124, 'asset_id': '1.3.5589'}, 
        # 'order_id': '1.7.563521855', 'account_id': '1.2.1819428', 
        # 'pays': {'amount': 1655, 'asset_id': '1.3.4157'}, 'receives': {'amount': 1405542, 'asset_id': '1.3.5589'}, 
        # 'fill_price': {'base': {'amount': 1672, 'asset_id': '1.3.4157'}, 'quote': {'amount': 1419979, 'asset_id': '1.3.5589'}}, 'is_maker': True}}

        # {'id': '5.0.198843620', 'key': {'base': '1.3.4157', 'quote': '1.3.5589', 'sequence': -691087}, 
        # 'time': '2025-04-20T20:42:30', 'op': {'fee': {'amount': 1124, 'asset_id': '1.3.5589'}, 
        # 'order_id': '1.7.563521855', 'account_id': '1.2.1819428', 
        # 'pays': {'amount': 1655, 'asset_id': '1.3.4157'}, 'receives': {'amount': 1405542, 'asset_id': '1.3.5589'}, 
        # 'fill_price': {'base': {'amount': 1672, 'asset_id': '1.3.4157'}, 'quote': {'amount': 1419979, 'asset_id': '1.3.5589'}}, 'is_maker': True}}

        if isinstance(info["live_trades"], list) and info["live_trades"]:
            if "timestamp" in info["live_trades"][0]:  # FIXME this check needs to be removed when bitshares is fixed

                live_trades = [
                    (unix_to_stamp(i["timestamp"] / 1000), i["price"], i["side"])
                    for i in info["live_trades"]
                ]
                # Traceback (most recent call last):
                #   File "/home/squid/Documents/QTRADEX/QTRADEX/EV_MAX/extinction_event_MAX3.py", line 542, in <module>
                #     main()
                #   File "/home/squid/Documents/QTRADEX/QTRADEX/EV_MAX/extinction_event_MAX3.py", line 534, in main
                #     qx.dispatch(bot, data, wallet)
                #   File "/home/squid/Documents/QTRADEX/QTRADEX/qtradex/qtradex/core/dispatch.py", line 113, in dispatch
                #     qx.core.live(bot, data, api_key, api_secret, dust, **kwargs)
                #   File "/home/squid/Documents/QTRADEX/QTRADEX/qtradex/qtradex/core/live.py", line 136, in live
                #     backtest(
                #   File "/home/squid/Documents/QTRADEX/QTRADEX/qtradex/qtradex/core/backtest.py", line 328, in backtest
                #     bot.plot(data, raw_states, indicator_states, block)
                #   File "/home/squid/Documents/QTRADEX/QTRADEX/EV_MAX/extinction_event_MAX3.py", line 412, in plot
                #     axes = qx.plot(
                #   File "/home/squid/Documents/QTRADEX/QTRADEX/qtradex/qtradex/plot/utilities.py", line 223, in plot
                #     live_trades = [
                #   File "/home/squid/Documents/QTRADEX/QTRADEX/qtradex/qtradex/plot/utilities.py", line 224, in <listcomp>
                #     (unix_to_stamp(i["timestamp"] / 1000), i["price"], i["side"])
                # TypeError: string indices must be integers

                buys = [(i[0], i[1]) for i in live_trades if i[2] == "buy"]
                sells = [(i[0], i[1]) for i in live_trades if i[2] == "sell"]
                if buys:
                    axes[0].scatter(*zip(*buys), c="yellow", marker="^", s=120)
                if sells:
                    axes[0].scatter(*zip(*sells), c="yellow", marker="v", s=120)

    # plot indicators
    plot_indicators(axes, timestamps, states, indicators, indicator_fmt)

    if len(states["trades"]) > 1:
        plot_trades(axes[0], states)

    start, end = axes[0].get_ylim()
    axes[0].set_yticks(np.exp(np.linspace(np.log(end), np.log(start), 10)))
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{sigfig(x, 3)}"))
    axes[0].yaxis.set_minor_formatter(plt.FuncFormatter(lambda x, _: f""))

    # Function to synchronize x-limits
    def sync_x_limits(event_ax):
        # Get the current x-limits of the event axis
        x_limits = event_ax.get_xlim()
        # Set the x-limits of the other axes
        for ax in axes[1:]:
            if ax != event_ax:
                ax.set_xlim(x_limits)

    axes[0].callbacks.connect("xlim_changed", lambda event: sync_x_limits(axes[0]))

    for i, ax in enumerate(axes[:]):  
        if i < len(axes) - 1:  # Check if it's not the last iteration
            ax.legend(loc="upper left", bbox_to_anchor=(0, 1))

        max_timestamp = max(timestamps).timestamp()
        min_timestamp = min(timestamps).timestamp()
        time_range = max_timestamp - min_timestamp
        pad = 0.005*time_range

        left = min_timestamp - pad
        right = max_timestamp + pad
        if int(left) == int(right):
            raise ValueError(f"Min and Max timestamps on axis {i+1} are identical, not plotting.")
        ax.set_xlim(
            unix_to_stamp(left),
            unix_to_stamp(right),
        )


    for ax in axes[:-1]:
        ax.set_xticks([])

    plot_balances(axes[-1], timestamps, states, data)
    maximize_window()

    def adjust(_):
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)

    plt.gcf().canvas.mpl_connect("resize_event", adjust)
    plotmotion(block)
    return axes


def plot_trades(axis, states):
    # Plot red/green win/loss lines between trade triangles
    p_op = states["trades"][0]
    for op in states["trades"][1:]:
        color = "lime" if op.profit >= 1 else "tomato"
        axis.plot(
            unix_to_stamp([p_op.unix, op.unix]),
            [p_op.price, op.price],
            color=color,
            linewidth=2,
        )
        p_op = op

    # Buy and Sell red/green trade triangle markers
    buys = list(
        zip(
            *[
                [unix_to_stamp(op.unix), op.price]
                for op in states["trades"]
                if isinstance(op, Buy)
            ]
        )
    )
    sells = list(
        zip(
            *[
                [unix_to_stamp(op.unix), op.price]
                for op in states["trades"]
                if isinstance(op, Sell)
            ]
        )
    )

    
    if buys:
        axis.scatter(*buys, c="lime", marker="^", s=80, zorder=999)
    if sells:
        axis.scatter(*sells, c="tomato", marker="v", s=80, zorder=999)
    # Bullseye on actual live trades per exchange API fill orders
    overrides = list(
        zip(
            *[
                [unix_to_stamp(op.unix), op.price]
                for op in states["trades"]
                if op.is_override
            ]
        )
    )
    if overrides:
        axis.scatter(*overrides, c="magenta", marker="o", s=120, zorder=996)
        axis.scatter(*overrides, c="yellow", marker="o", s=60, zorder=997)



def plot_balances(axis, timestamps, states, data):
    # plot balances chart
    balances = rotate(states["balances"])

    (
        balances[data.asset],
        balances[data.currency],
    ) = compute_potential_balances(
        balances[data.asset],
        balances[data.currency],
        states["close"],
    )

    ax = None
    lines = []
    for idx, (token, balance) in list(enumerate(balances.items())):
        # Handle parasite axes
        if ax is None:
            ax = axis
        else:
            ax = ax.twinx()
        # Draw red/green lines between drawdown extremes
        max_x = timestamps[np.argmax(balance)]
        max_y = np.max(balance)
        min_x = timestamps[np.argmin(balance)]
        min_y = np.min(balance)
        start_x = timestamps[0]
        start_y = balance[0]
        end_x = timestamps[-1]
        end_y = balance[-1]
        if min_x < max_x:
            ax.plot([start_x, min_x], [start_y, min_y], color="tomato") 
            ax.plot([min_x, max_x], [min_y, max_y], color="lime")
            ax.plot([max_x, end_x], [max_y, end_y], color="tomato")
        else:
            ax.plot([start_x, max_x], [start_y, max_y], color="lime") 
            ax.plot([max_x, min_x], [max_y, min_y], color="tomato")
            ax.plot([min_x, end_x], [min_y, end_y], color="lime")
        # Vertical drawdown
        ax.plot([max_x, max_x], [max_y, balance[0]], color="yellow", linestyle='--', zorder=999)
        ax.plot([min_x, min_x], [min_y, balance[0]], color="yellow", linestyle='--', zorder=999)
        ax.plot([end_x, end_x], [end_y, balance[0]], color="yellow", zorder=999)
        # Add balance lines
        color = ["mediumorchid", "deepskyblue", "lime"][idx % 3]
        lines.append(
            ax.step(timestamps, balance, label=token, color=color, where="post")[0]
        )
        # Fill the area above/below the initial balance in green/red
        ax.fill_between(timestamps, balance, start_y, where=(balance > start_y), 
                        color='lime', alpha=0.15, step="post")
        ax.fill_between(timestamps, balance, start_y, where=(balance < start_y), 
                        color='tomato', alpha=0.3, step="post")
        # Plot a dashed horizontal initial balance line
        ax.axhline(balance[0], color=color,alpha=0.25, linestyle='--')
        # Plot yellow dots at max drawdown / max gain and start/stop
        ax.scatter([start_x, min_x, max_x, end_x], [start_y, min_y, max_y, end_y], color="yellow", s=80, marker='o', zorder=999)
        # Plot log scale and set tick label colors to match respective balance line
        ax.set_yscale("log")
        ax.tick_params(axis="y", which="both", labelcolor=color)

    # Create legend
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels)
    # Add the word "Balances" to the plot
    ax.text(
        0.5,
        0.15,
        "Balances",
        transform=ax.transAxes,
        color="white",
        alpha=0.15,
        ha="center",
        va="center",
        zorder=-1,
        fontdict={"fontsize": 40, "fontweight": "bold", "family": "monospace"},
    )


def compute_potential_balances(asset_balance, currency_balance, price):
    # Convert inputs to numpy arrays for efficient computation
    asset_balance = np.array(asset_balance)
    currency_balance = np.array(currency_balance)
    price = np.array(price)

    # Calculate the potential asset balance if all currency were sold at current price
    potential_assets = currency_balance / price

    # Calculate the potential currency if all assets were sold at current price
    potential_currency = asset_balance * price

    # Merge the actual currency with potential currency
    merged_currency_balance = np.where(
        currency_balance > NIL, currency_balance, potential_currency
    )
    # Repeat for assets
    merged_asset_balance = np.where(
        asset_balance > NIL, asset_balance, potential_assets
    )

    return merged_asset_balance, merged_currency_balance
