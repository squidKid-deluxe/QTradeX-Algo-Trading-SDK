import json
import time
from pprint import pprint

import numpy as np
import qtradex as qx
from qtradex.common.utilities import it, rotate, sigfig
from qtradex.core.base_bot import Info
from qtradex.core.quant import preprocess_states, slice_candles
from qtradex.core.ui_utilities import logo
from qtradex.private.signals import Buy, Hold, Sell, Thresholds
from qtradex.private.wallet import PaperWallet


def trade(asset, currency, operation, wallet, price, now):
    """
    Execute a trade operation (buy or sell) based on the current market price and wallet balance.

    Parameters:
    - asset: The asset to trade.
    - currency: The currency to trade with.
    - operation: The trading operation (Buy or Sell).
    - wallet: The user's wallet containing asset and currency balances.
    - price: The current market price data.
    - now: The current time in UNIX format.

    Returns:
    - Updated wallet and the operation performed (if any).
    """
    initial_value = wallet.value((asset, currency))
    execution = price["close"]

    # Determine execution price based on operation type and market conditions
    execution, operation = determine_execution_price(
        operation, wallet, price, execution, asset, currency
    )

    # Ensure execution price is within market bounds
    execution = min(max(execution, price["low"]), price["high"])

    # Perform the trade operation (buy or sell)
    wallet, operation = perform_trade(operation, wallet, asset, currency, execution)

    # Update operation details if a trade was made
    if isinstance(operation, (Buy, Sell)):
        operation.price = execution
        operation.unix = now
        operation.profit = wallet.value((asset, currency), execution) / initial_value
    else:
        operation = None

    return wallet, operation


def determine_execution_price(operation, wallet, price, execution, asset, currency):
    """
    Determine the execution price for the trade based on the operation type and market conditions.

    Parameters:
    - operation: The trading operation (Buy or Sell).
    - wallet: The user's wallet containing asset and currency balances.
    - price: The current market price data.
    - execution: The initial execution price.

    Returns:
    - Updated execution price and operation.
    """
    low, high = sorted([price["open"], price["close"]])
    low = (low + price["low"])/2
    high = (high + price["high"])/2
    # low, high = sorted([price["low"], price["high"]])

    if isinstance(operation, Thresholds):
        if wallet[asset]:
            if high > operation.selling:
                execution = operation.selling
                operation = Sell(maxvolume=operation.maxvolume)
                operation.is_override = False
        elif wallet[currency]:
            if low < operation.buying:
                execution = operation.buying
                operation = Buy(maxvolume=operation.maxvolume)
                operation.is_override = False
    elif operation.price is not None:
        if wallet[asset] and isinstance(operation, Sell):
            if high > operation.price:
                execution = operation.price
            else:
                operation = None
        elif wallet[currency] and isinstance(operation, Buy):
            if low < operation.price:
                execution = operation.price
            else:
                operation = None

    return execution, operation


def perform_trade(operation, wallet, asset, currency, execution):
    """
    Execute the trade operation (buy or sell) and update the wallet accordingly.

    Parameters:
    - operation: The trading operation (Buy or Sell).
    - wallet: The user's wallet containing asset and currency balances.
    - asset: The asset to trade.
    - currency: The currency to trade with.
    - execution: The execution price for the trade.

    Returns:
    - Updated wallet and operation.
    """
    if isinstance(operation, Buy):
        volume = min(wallet[currency], operation.maxvolume)
        if not volume:
            return wallet, None
        wallet[asset] += (volume / execution) * (1-wallet.fee/100)
        wallet[currency] -= volume

    elif isinstance(operation, Sell):
        volume = min(wallet[asset], operation.maxvolume)
        if not volume:
            return wallet, None
        wallet[asset] -= volume
        wallet[currency] += (volume * execution) * (1-wallet.fee/100)

    return wallet, operation


def backtest(
    bot,
    data,
    wallet=None,
    plot=True,
    block=True,
    return_states=False,
    range_periods=True,
    show=True,
    fine_data=None,
    always_trade="smart",
):
    """
    Run a backtest for the trading bot using historical data.

    Parameters:
    - bot: The trading bot instance.
    - data: Historical market data.
    - wallet: Optional initial wallet state.
    - plot: Whether to plot results.
    - block: Whether to block execution during plotting.
    - return_states: Whether to return the raw states.
    - range_periods: Whether to adjust tuning parameters based on candle size.
    - show: Whether to display results in the console.
    - fine_data: Optional fine-grained data for more precise trading.
    - always_trade: Whether to allow trading on every tick.

    Returns:
    - A dictionary containing the results of the backtest, including performance metrics.
    """
    if show and plot:
        logo(animate=False)

    # Initialize bot info if not already set
    if not hasattr(bot, "info"):
        bot.info = Info({"mode": "backtest"})
        # This line is for debug purposes, it creates a "live" style plot during a backtest
        # bot.info = Info({"mode": "live", "live_data":data.fine_data, "start":data.end})

    # Initialize wallet if not provided
    if wallet is None:
        wallet = PaperWallet({data.asset: 0, data.currency: 1})

    bot.reset()
    begin = data.begin
    end = data.end
    days = (end - begin) / 86400
    candle_size = data.candle_size
    warmup = bot.autorange()

    orig_tune = bot.tune.copy()

    # Adjust tuning parameters based on candle size if required
    if range_periods:
        adjust_tuning_parameters(bot, candle_size)

    now = begin + (candle_size * (warmup + 1))
    initial_data = slice_candles(now, data, candle_size, 1)

    # Set initial wallet value based on the initial market price
    wallet.value((data.asset, data.currency), initial_data["close"])

    indicator_states = []
    states = []
    indicators = bot.indicators(data)

    # Ensure all indicators are of the same length
    if indicators:
        minlen = min(map(len, indicators.values()))
        indicators = {k: v[-minlen:] for k, v in indicators.items()}
        indicated_data = {"indicators": rotate(indicators)}
    else:
        minlen = min(map(len, data.values()))
        indicated_data = {"indicators": [None for _ in range(minlen)]}
    indicated_data.update({k: v[-minlen:] for k, v in data.items()})
    last_trade = None

    # Record initial state if the current time is beyond the end of the data
    if now > end:
        states.append(
            {"trades": None, "balances": wallet.copy(), "unix": now, **initial_data}
        )

    last_trade_time = 0

    ticks = 0
    if fine_data is None and data.fine_data is not None:
        fine_data = data.fine_data

    # Main backtesting loop
    while now <= end:
        tickdx = np.searchsorted(indicated_data["unix"], now, side="left")
        # this "fast forward" is largely for weekends, though
        # it works for other types of missing data
        try:
            tick_data = {k: v[tickdx] for k, v in indicated_data.items()}
            if abs(tick_data["unix"] - now) > data.candle_size:
                now += candle_size
                continue
        except IndexError:
            now += candle_size
            continue  # Skip to the next time step if tick data is not available

        fine_tick_data = get_fine_tick_data(data, fine_data, now)#-candle_size)
        # Protect the wallet from accidental modifications
        wallet._protect()
        indicators = tick_data["indicators"]
        operation = bot.strategy(
            {"last_trade": last_trade, "unix": now, "wallet": wallet, **tick_data},
            indicators,
        )

        # Check if enough time has passed to trade again
        if (

            # If the elapsed time since last trade 
            # is greater than the largest candle size in the dataset
            (now - last_trade_time >= data.base_size)
            

            # we can also pass "always_trade" as a kwarg to backtest()
            or (always_trade is True)


            # Smart mode means that the elapsed time is greater than the candle size we're using during this moment of the backtest.  
            # That is... some backtests we use finer grain data with daily high/low.
            # In the case of "we're currently using daily candles"; then only trade once daily
            
            or (
                always_trade == "smart"
                and (now - last_trade_time) >= fine_tick_data["candle_size"]
            )
        ) and not isinstance(operation, Hold):
            if operation is not None:
                wallet._release()  # Release write protection to perform trade
                wallet, operation = trade(
                    data.asset, data.currency, operation, wallet, fine_tick_data, now
                )
                last_trade_time = now

                # Store the last trade operation
                if operation is not None:
                    last_trade = operation
        else:
            operation = None  # No trade executed

        # Record the current state
        states.append(
            {
                "trades": operation,
                "balances": wallet.copy(),
                "unix": now,
                **tick_data,
            }
        )
        indicator_states.append(indicators)
        now += candle_size
        ticks += 1

    # Process and rotate states for output
    states = rotate(states)
    states["trades"] = [i for i in states["trades"] if i is not None]
    indicator_states = rotate(indicator_states)
    states["indicator_states"] = indicator_states

    # Extract trade times and prices for analysis
    if states["trades"]:
        states["trade_times"], states["trade_prices"] = list(
            zip(*[[op.unix, op.price] for op in states["trades"]])
        )
    else:
        states["trade_times"], states["trade_prices"] = [], []

    states["trade_colors"] = [
        "green" if isinstance(i, Buy) else "red" for i in states["trades"]
    ]

    raw_states = states
    states = preprocess_states(states, (data.asset, data.currency))
    states["days"] = days
    states["candle_size"] = candle_size
    states["begin"] = raw_states["unix"][0]
    states["end"] = raw_states["unix"][-1]

    # Calculate fitness metrics
    keys, custom = bot.fitness(states, raw_states, data.asset, data.currency)
    if "roi" not in keys:
        keys.append("roi")

    bot.tune = orig_tune

    # Calculate the final results of the backtest
    ret = {
        **qx.indicators.fitness.fitness(
            keys, states, raw_states, data.asset, data.currency
        ),
        **custom,
    }

    # I don't care how good the results are, if you don't make at least some trades, you don't count
    if len(raw_states["trades"]) < 10:
        ret = {k:v - 10000 for k, v in ret.items()}


    # Plot results if requested
    if plot:
        if show:
            print_backtest_results(bot, states, data, ret, ticks, candle_size)
        bot.plot(data, raw_states, indicator_states, block)

    # If requested, return the raw states along with the fitness metrics
    if return_states:
        ret = [ret, raw_states, states]

    return ret


def print_backtest_results(bot, states, data, ret, ticks, candle_size):
    pprint(bot.tune, indent=4)
    for op in states["detailed_trades"]:
        obj = op["object"]
        direction = "BUY " if isinstance(obj, Buy) else "SELL"
        reason = f"  {obj.reason}" if getattr(obj, "reason", None) else ""
        sign = "+" if op["roi"] >= 1 else ""
        color = "green" if op["roi"] >= 1 else "red"
        print(
            f'[{time.ctime(op["unix"])}]',
            " ",
            direction,
            it(color, f'{sign}{sigfig((op["roi"]-1)*100, 6):.1f}%'.ljust(8, "0")),
            reason,
        )
    print(json.dumps(ret, indent=4))
    print(
        f"Days: {data.days:.2f}   Ticks: {ticks}   "
        f"Days per trade: {(ticks*candle_size) / ((len(states['detailed_trades']) + 1))/86400:.2f}"
    )
    print(it("yellow", f'{bot.info["mode"].upper()} TRADING AT {data.exchange.upper()}'))


def adjust_tuning_parameters(bot, candle_size):
    """
    Adjust the bot's tuning parameters based on the candle size.

    Parameters:
    - bot: The trading bot instance.
    - candle_size: The size of the candles in seconds.
    """
    for k, v in list(bot.tune.items()):
        if k.endswith("_period") or k.endswith("_periods"):
            if isinstance(v, float):
                bot.tune[k] = v * (86400 / candle_size)
            elif isinstance(v, int):
                bot.tune[k] = int(v * (86400 / candle_size))
            elif isinstance(v, np.ndarray):
                bot.tune[k] = (v * (86400 / candle_size)).astype(v.dtype)


def get_fine_tick_data(data, fine_data, now):
    """
    Retrieve fine-grained tick data if available, otherwise return the regular tick data.

    Parameters:
    - fine_data: Optional fine-grained data for more precise trading.
    - now: The current time in UNIX format.

    Returns:
    - The fine tick data if available, otherwise the regular tick data.
    """
    if fine_data is None:
        fine_data = data
    fine_tickdx = np.searchsorted(fine_data["unix"], now, side="left")
    return {k: v[fine_tickdx] for k, v in fine_data.items()}
