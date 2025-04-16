import json
import time

import numpy as np
import qtradex as qx
from qtradex.common.utilities import it, rotate, sigfig
from qtradex.core.base_bot import Info
from qtradex.core.quant import preprocess_states, slice_candles
from qtradex.core.ui_utilities import logo
from qtradex.private.signals import Buy, Sell, Thresholds
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
        wallet[asset] += volume / execution
        wallet[currency] -= volume

    elif isinstance(operation, Sell):
        volume = min(wallet[asset], operation.maxvolume)
        if not volume:
            return wallet, None
        wallet[asset] -= volume
        wallet[currency] += volume * execution

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
    minlen = min(map(len, indicators.values()))
    indicators = {k: v[-minlen:] for k, v in indicators.items()}
    indicated_data = {"indicators": rotate(indicators)}
    indicated_data.update({k: v[-minlen:] for k, v in data.items()})
    last_trade = None

    # Record initial state if the current time is beyond the end of the data
    if now > end:
        states.append(
            {"trades": None, "balances": wallet.copy(), "unix": now, **initial_data}
        )

    last_trade_time = 0

    ticks = 0

    # Main backtesting loop
    while now <= end:
        tickdx = np.searchsorted(indicated_data["unix"], now, side="left")
        try:
            tick_data = {k: v[tickdx] for k, v in indicated_data.items()}
        except IndexError:
            now += candle_size
            continue  # Skip to the next time step if tick data is not available

        fine_tick_data = get_fine_tick_data(data, fine_data, now, tick_data)

        # Protect the wallet from accidental modifications
        wallet._protect()
        indicators = tick_data["indicators"]
        operation = bot.strategy(
            {"last_trade": last_trade, "unix": now, "wallet": wallet, **tick_data},
            indicators,
        )

        # Store the last trade operation
        if operation is not None:
            last_trade = operation

        # Check if enough time has passed to trade again
        if (
            (now - last_trade_time >= data.base_size)
            or (always_trade and isinstance(always_trade, bool))
            or (
                always_trade == "smart"
                and (now - last_trade_time) >= fine_tick_data["candle_size"]
            )
        ):
            if operation is not None:
                wallet._release()  # Release write protection to perform trade
                wallet, operation = trade(
                    data.asset, data.currency, operation, wallet, fine_tick_data, now
                )
                last_trade_time = now
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

    # Plot results if requested
    if plot:
        if show:
            print(json.dumps(bot.tune, indent=4))
            for op in states["detailed_trades"]:
                if op["roi"] >= 1:
                    print(
                        f'[{time.ctime(op["unix"])}]',
                        " ",
                        "BUY " if isinstance(op["object"], Buy) else "SELL",
                        " ",
                        it("green", f'{sigfig((op["roi"]-1)*100, 6):.1f}'.ljust(4, "0")
                        + "% GAIN")
                    )
                else:
                    print(
                        f'[{time.ctime(op["unix"])}]',
                        " ",
                        "BUY " if isinstance(op["object"], Buy) else "SELL",
                        " ",
                        it("red", f'{sigfig((1-op["roi"])*100, 6):.1f}'.ljust(4, "0")
                        + "% LOSS")
                    )
            print(json.dumps(ret, indent=4))
            print(
                f"Days: {data.days:.2f}   Ticks: {ticks}   "
                f"Days per trade: {(ticks*candle_size) / ((len(states['detailed_trades']) + 1))/86400:.2f}"
            )
            print(it("yellow", f'{bot.info["mode"].upper()} TRADING AT {data.exchange.upper()}'))
        bot.plot(data, raw_states, indicator_states, block)

    # If requested, return the raw states along with the fitness metrics
    if return_states:
        ret = [ret, raw_states, states]

    return ret


def adjust_tuning_parameters(bot, candle_size):
    """
    Adjust the bot's tuning parameters based on the candle size.

    Parameters:
    - bot: The trading bot instance.
    - candle_size: The size of the candles in seconds.
    """
    for k, v in list(bot.tune.items()):
        if k.endswith("_period"):
            if isinstance(v, float):
                bot.tune[k] = v * (86400 / candle_size)
            else:
                bot.tune[k] = int(v * (86400 / candle_size))


def get_fine_tick_data(data, fine_data, now, tick_data):
    """
    Retrieve fine-grained tick data if available, otherwise return the regular tick data.

    Parameters:
    - fine_data: Optional fine-grained data for more precise trading.
    - now: The current time in UNIX format.
    - tick_data: The regular tick data.

    Returns:
    - The fine tick data if available, otherwise the regular tick data.
    """
    if fine_data is not None:
        fine_tickdx = np.searchsorted(fine_data["unix"], now, side="left")
        return {k: v[fine_tickdx] for k, v in fine_data.items()}
    return {**tick_data, "candle_size": data.candle_size}
