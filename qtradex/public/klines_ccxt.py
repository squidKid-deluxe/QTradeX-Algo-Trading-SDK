import math
import time

import ccxt
import numpy as np
from qtradex.common.utilities import (format_timeframe, rotate, to_iso_date,
                                      trace, unformat_timeframe)
from qtradex.public.utilities import BadTimeframeError, clip_to_time_range

DETAIL = False
ATTEMPTS = 5


def klines_ccxt(exchange, asset, currency, start, end, interval):
    """
    Input and output normalized requests for candle data.
    Returns a dict with numpy array values for the following keys:
    ["high", "low", "open", "close", "volume", "unix"]
    where unix is int and the remainder are float.
    This is the ideal format for utilizing talib / tulip indicators.
    """

    api = {
        "exchange": exchange,
        "pair": f"{asset}/{currency}",
        "ccxt_hook": getattr(ccxt, exchange)(),
    }
    if not api["ccxt_hook"].has["fetchOHLCV"]:
        raise ValueError(f"{exchange} does not support klines.")
    if api["pair"] not in api["ccxt_hook"].load_markets().keys():
        raise ValueError(api["pair"])

    if end is None:
        # to current
        end = int(time.time())
    if start is None:
        # default 10 candles
        start = end - 10 * interval
    # allow for timestamp up to one interval and one minute in future.
    end = end + interval + 60
    # request 3 candles deeper than needed
    deep_begin = start - 3 * interval
    if DETAIL:
        print("\nstart:", to_iso_date(start), "end:", to_iso_date(end))

    idx = 0

    while True:
        idx += 1
        try:
            # Collect external data in pages if need be
            data = paginate_candles(api, deep_begin, end, interval)
            if DETAIL:
                print(len(data), "paginated with overlap and collated")
            data = remove_null(data)
            if DETAIL:
                print(len(data), "null data removed")
            data = no_duplicates(data)
            if DETAIL:
                print(len(data), "edge match - no duplicates by unix")
            data = sort_by_unix(data)
            if DETAIL:
                print(len(data), "edge match - sort by unix")
            data = rotate(data)
            if DETAIL:
                print(
                    len(data["unix"]),
                    data["unix"][0],
                    data["unix"][-1],
                    "rotation; reformatted to dict of lists",
                )
            data = interpolate_previous(data, deep_begin, end, interval)
            if DETAIL:
                print(
                    len(data["unix"]),
                    len(data["close"]),
                    "missing buckets to candles interpolated as previous close",
                )
            data = {k: np.array(v) for k, v in data.items()}
            # data = clip_to_time_range(data, start, end)
            # if DETAIL:
            #     print(
            #         len(data["unix"]),
            #         "windowed to initial start / end request",
            #     )
            data = left_strip(data)
            if DETAIL:
                print(
                    len(data["unix"]),
                    "stripped of empty pre-market candles",
                )
            data = normalize(data)
            if DETAIL:
                print({k: len(v) for k, v in data.items()})
            if DETAIL:
                print("normalized as valid: high is highest, no extremes, etc.")
            if DETAIL:
                print("final conversion to dict of numpy arrays:\n")
                print(
                    "total items",
                    len(data),
                    "/",
                    len(data["unix"]),
                    "keys",
                    data.keys(),
                    "type",
                    type(data["unix"]),
                )
                print("\n\nRETURNING", exchange.upper(), api["pair"], "CANDLE DATA\n\n")
            return data
        except BadTimeframeError:
            raise
        except Exception as error:
            if idx > ATTEMPTS:
                raise
            msg = trace(error)
            print(msg, {k: v for k, v in api.items() if k != "secret"})
            continue


# Pagination function
def paginate_candles(api, start, end, interval):
    """
    Paginate requests per maximum request size per exchange.
    Collate responses crudely with overlap.
    Previously, this used a `max_candles` dictionary that gave the maximum number
    of candles each exchange would return at once. This is now determined
    empirically at runtime.

    Note that `candles` (aka ccxt) doesn't provide for an end time. Thus, this
    simply queries until enough candles are gathered.
    """
    overlap = 2

    # Determine number of candles we require
    depth = int(math.ceil((end - start) / float(interval)))
    # Attempt to gather all the candles at once
    data = []
    last_chunk = candles(api, start, interval, limit=depth)
    data.extend(last_chunk)
    depth -= len(last_chunk)
    # If that didn't return enough
    while depth > 0 and last_chunk:
        # Find how many are left
        print(f"Only got {len(last_chunk)} datapoints, {depth} left.")
        # Tick forward, but leave an `overlap`
        start += (len(last_chunk) - overlap) * interval
        # Don't anger the database overlords
        time.sleep(2)
        # Get more candles
        last_chunk = candles(api, start, interval, limit=depth)
        data.extend(last_chunk)
        depth -= len(last_chunk)
    return data


def candles(api, start, interval, limit):
    while True:
        try:
            # initialize the exchange
            exchange = api["ccxt_hook"]
            timeframe = format_timeframe(interval)
            print(f"Collecting {limit} candles for {api['pair']} from {start} @ {timeframe} candles")
            if timeframe not in exchange.timeframes:
                raise BadTimeframeError(
                    "Valid timeframes: ",
                    [unformat_timeframe(i) for i in exchange.timeframes.keys()],
                )
            page = exchange.fetch_ohlcv(
                symbol=api["pair"], timeframe=timeframe, since=int(start * 1000), limit=limit
            )
            return [
                {
                    "unix": i[0] / 1000,
                    **dict(zip(["open", "high", "low", "close", "volume"], i[1:])),
                }
                for i in page
            ]
        except (
            ccxt.DDoSProtection,
            ccxt.ExchangeNotAvailable,
            ccxt.ExchangeError,
            ccxt.InvalidNonce,
        ):
            print("Encountered rate limit!  Pausing for 10 seconds...")
            time.sleep(10)
            continue


# Remove null data function
def remove_null(data):
    """
    Ensure all data in list are dicts with a "unix" key
    """
    return [i for i in data if isinstance(i, dict) and "unix" in i]


# Remove duplicates function
def no_duplicates(data):
    """
    Ensure no duplicates due to pagination overlap at edges
    """
    dup_free = []
    timestamps = []
    for item in data:
        if item["unix"] not in timestamps:
            timestamps.append(item["unix"])
            dup_free.append(item)

    return dup_free


# Sort by unix timestamp function
def sort_by_unix(data):
    """
    Pagination may still be backwards and segmented; resort by timestamp
    """
    return sorted(data, key=lambda k: k["unix"])


# Interpolate previous function
def interpolate_previous(data, start, end, interval):
    """
    Candles may be missing; fill them in with previous close
    """
    start = int(start)
    end = int(end)
    interval = int(interval)
    ip_unix = list(range(int(min(data["unix"])), int(max(data["unix"])), int(interval)))
    out = {
        "high": [],
        "low": [],
        "open": [],
        "close": [],
        "volume": [],
        "unix": ip_unix,
    }
    for candle in ip_unix:
        match = False
        for idx, _ in enumerate(data["unix"]):
            diff = candle - data["unix"][idx]
            if 0 <= diff < interval:
                match = True
                out["volume"].append(data["volume"][idx])
                out["high"].append(data["high"][idx])
                out["low"].append(data["low"][idx])
                out["open"].append(data["open"][idx])
                out["close"].append(data["close"][idx])
                break

        if not match:
            if candle == start:
                close = data["close"][0]
            else:
                close = out["close"][-1]
            out["volume"].append(0)
            out["high"].append(close)
            out["low"].append(close)
            out["open"].append(close)
            out["close"].append(close)

    return out


# Left strip function
def left_strip(data):
    """
    Remove no volume candles from the beginning of the dataset.
    """
    first_non_zero_index = next(
        (i for i, v in enumerate(data["volume"]) if v > 0), len(data["volume"])
    )
    return {
        "high": data["high"][first_non_zero_index:],
        "low": data["low"][first_non_zero_index:],
        "open": data["open"][first_non_zero_index:],
        "close": data["close"][first_non_zero_index:],
        "volume": data["volume"][first_non_zero_index:],
        "unix": data["unix"][first_non_zero_index:],
    }


# Normalize function
def normalize(data):
    """
    Ensure high is high and low is low.
    Filter extreme candlesticks at 0.5X to 2X the candlestick average.
    Ensure open and close are within high and low.
    """
    for i in range(len(data["close"])):
        data["high"][i] = max(
            data["high"][i], data["low"][i], data["open"][i], data["close"][i]
        )
        data["low"][i] = min(
            data["high"][i], data["low"][i], data["open"][i], data["close"][i]
        )
        ocl = (data["open"][i] + data["close"][i] + data["low"][i]) / 3
        och = (data["open"][i] + data["close"][i] + data["high"][i]) / 3
        data["high"][i] = min(data["high"][i], 2 * ocl)
        data["low"][i] = max(data["low"][i], och / 2)
        data["open"][i] = min(data["open"][i], data["high"][i])
        data["open"][i] = max(data["open"][i], data["low"][i])
        data["close"][i] = min(data["close"][i], data["high"][i])
        data["close"][i] = max(data["close"][i], data["low"][i])

    return data
