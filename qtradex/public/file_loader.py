import hashlib
import json
from typing import Optional

import jsonpickle
import numpy as np
from qtradex.common.utilities import (PATH, json_ipc, parse_date, read_file,
                                      write_file)
from qtradex.public.data import Data
from qtradex.public.utilities import clip_to_time_range, reaggregate
import time
import csv

DETAIL = True


def dprint(*args, **kwargs):
    if DETAIL:
        print(*args, **kwargs)


def load_csv(
    exchange: str,
    asset: str,
    currency: str,
    filepath: str,
    begin: Optional[str] = None,
    end: Optional[str] = None,
    candle_size: int = 86400,  # Default to daily candles, as in load_csv
    stride: int = None,
) -> Data:
    """
    Retrieves and caches candlestick data from a CSV file. If the data is already cached for the given parameters,
    it returns the cached version. Otherwise, it loads the data using load_csv, caches it, and returns it.

    This function uses a unique key and handles persistence via JSON files, similar to the original caching mechanism.

    Args:
    - exchange: The exchange name (e.g., 'kraken').
    - asset: The asset being traded (e.g., 'BTC').
    - currency: The currency (e.g., 'USD').
    - filepath: Path to the CSV file.
    - begin: Start time for data (e.g., '2019-07-01').
    - end: End time for data (e.g., '2020-01-01').
    - candle_size: The size of each candle in seconds.

    Returns:
    - A Data object containing the candlestick data.

    Raises:
    - FileNotFoundError: If the CSV file or cache files are inaccessible.
    - ValueError: If there's an issue with parameters or serialization.
    """
    begin, end = parse_date(begin), parse_date(end)
    # Generate a unique index key, similar to the original function
    index_key = hashlib.md5(
        str((exchange, asset, currency, candle_size, stride, filepath)).encode()
    ).hexdigest()

    # Load or initialize the data index
    try:
        index = json_ipc("data_index.json")  # Rely on json_ipc to handle the path
    except FileNotFoundError:
        json_ipc("data_index.json", "{}")  # Create an empty index
        index = " {}\n"
    total_time = [begin, end]  # Track the requested time range

    if index_key in index:
        # Cached data exists; check if it covers the requested range
        cached_range = index[index_key]  # e.g., [start_timestamp, end_timestamp]
        if ranges_overlap(cached_range, [begin, end]):
            # Return cached data if there's overlap
            data = load_from_cache(index_key)
            data.raw_candles = clip_to_time_range(data.raw_candles, begin, end)
            data.fine_data = clip_to_time_range(data.fine_data, begin, end)
            data.begin = begin
            data.end = end
            data.days = (end - begin) / 86400
            return data
        else:
            # Cached range doesn't fully cover; need to reload
            print("Partial overlap detected; reloading for full range.")

    # No cache or needs update: Load fresh data
    import_data = load_fresh_data(
        exchange, asset, currency, filepath, begin, end, candle_size, stride
    )

    # Cache the data
    cache_data(index_key, import_data, total_time)

    return import_data  # Return the freshly loaded Data object


def load_fresh_data(
    exchange: str,
    asset: str,
    currency: str,
    filepath: str,
    begin=None,
    end=None,
    candle_size=86400,
    stride=None,  # Optional: stride for candles; defaults to candle_size if not provided
    keys=("unix", "price", "volume"),
) -> Data:
    if stride is None:
        stride = candle_size
        hold_stride = stride
        hold_candle_size = candle_size
    else:
        hold_stride = stride
        hold_candle_size = candle_size
        candle_size = stride

    keydxes = {key: keys.index(key) for key in keys}

    keys = list(keys)
    use_discrete = not all(key in keys for key in ["high", "low", "open", "close"])

    # Parse dates
    if begin is not None:
        begin = parse_date(begin)
    else:
        begin = float("-inf")
    if end is not None:
        end = parse_date(end)
    else:
        end = float("inf")

    unix_dx = next((i for i, key in enumerate(keys) if "unix" in key), None)
    if unix_dx is not None:
        multiplier = (
            1e3
            if keys[unix_dx] == "unix_milli"
            else 1e6
            if keys[unix_dx] == "unix_micro"
            else 1
        )
        keys[unix_dx] = "unix"
    else:
        raise ValueError("No timestamp key found in keys")

    dprint("Reading and aggregating data into candles...\n")
    candles = []  # List to hold finalized candles
    current_candle = None  # Will hold the state of the current candle
    last_printed = time.time()
    with open(filepath, mode="r", newline="") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            try:
                ctime = float(row[unix_dx]) / multiplier  # Convert to Unix seconds
            except (IndexError, ValueError):
                continue  # Skip malformed rows

            if ctime < begin:
                continue  # Skip before start
            if ctime > end:
                dprint("Found timestamp past end date; stopping read.")
                break  # Assume ordered data

            if time.time() - last_printed > 0.2:
                dprint(f"\033[AProcessing: {time.ctime(ctime)}")
                last_printed = time.time()

            # Determine the candle's start time for this row
            candle_start = (
                ctime // stride * stride
            )  # Floor to the nearest stride boundary

            if (
                current_candle is None
                or ctime < current_candle["unix"]
                or ctime >= current_candle["unix"] + candle_size
            ):
                # Finalize the previous candle if it exists
                if current_candle is not None:
                    candles.append(current_candle)
                # Start a new candle
                if use_discrete:
                    price = float(row[keydxes["price"]])
                    current_candle = {
                        "unix": candle_start,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": 0,
                    }
                else:
                    current_candle = {
                        "unix": candle_start,
                        "open": float(row[keydxes["open"]]),
                        "high": float(row[keydxes["high"]]),
                        "low": float(row[keydxes["low"]]),
                        "close": float(row[keydxes["close"]]),
                        "volume": 0,
                    }
                if "volume" in keys:
                    current_candle["volume"] += float(row[keydxes["volume"]])
                else:
                    current_candle["volume"] = 1

            # Update the current candle with this row's data
            if current_candle and candle_start == current_candle["unix"]:
                if use_discrete:
                    prices = (float(row[keydxes["price"]]),)
                else:
                    prices = (
                        float(row[keydxes["open"]]),
                        float(row[keydxes["high"]]),
                        float(row[keydxes["low"]]),
                        float(row[keydxes["close"]]),
                    )
                current_candle["high"] = max([current_candle["high"], *prices])
                current_candle["low"] = min([current_candle["low"], *prices])
                current_candle["close"] = prices[-1]
                if "volume" in keys:
                    current_candle["volume"] += float(row[keydxes["volume"]])

        # After the loop, add the last candle if it exists
        if current_candle is not None:
            candles.append(current_candle)

    # Now, candles is a list of dictionaries; convert to NumPy arrays
    if candles:
        data = {k: np.array([candle[k] for candle in candles]) for k in candles[0]}

        # Check and reaggregate if needed, as in original
        if len(data.get("unix", [])) > 1:
            empirical_size = data["unix"][1] - data["unix"][0]
            if abs(empirical_size - candle_size) >= 1:
                dprint("Reaggregating candles...")
                data = reaggregate(data, candle_size)
    else:
        data = {}  # Empty data if no candles were created

    # Ensure volume is present
    if "volume" not in data:
        data["volume"] = np.ones_like(data.get("unix", np.array([])))

    data = {k: np.array(v) for k, v in data.items()}  # Final conversion to numpy arrays
    data["candle_size"] = np.full(data["unix"].shape, hold_candle_size)

    if hold_stride != hold_candle_size:
        reagg_data = reaggregate(data, hold_candle_size, hold_stride)
    else:
        reagg_data = data

    # Create and return the Data object
    data_class = Data(
        exchange,
        asset,
        currency,
        begin=min(reagg_data.get("unix", [float("inf")])),
        end=max(reagg_data.get("unix", [float("-inf")])),
        candle_size=candle_size,
        placeholder=True,
    )
    data_class.raw_candles = reagg_data
    data["candle_size"] = np.full_like(data["unix"], hold_stride)
    data_class.fine_data = data
    data_class.base_size = hold_candle_size
    data_class.candle_size = hold_stride
    return data_class


def cache_data(index_key: str, data: Data, total_time: list):
    """Serialize and cache the Data object."""
    # Convert Data object to a serializable dictionary
    cache_dict = jsonpickle.encode(data)

    cache_file = (
        PATH + "pipe/" + f"{index_key}_csv_candles.json"
    )  # json_ipc will handle the full path
    write_file(cache_file, cache_dict)  # Save to JSON

    # Update the index with the time range
    index = json_ipc("data_index.json")
    index[index_key] = total_time  # Store as [begin, end]
    json_ipc("data_index.json", json.dumps(index))


def load_from_cache(index_key: str) -> Data:
    """Load and reconstruct a Data object from the cache."""
    cache_file = f"{index_key}_csv_candles.json"
    cache_file = PATH + "pipe/" + cache_file
    cached_data = jsonpickle.decode(json.loads(read_file(cache_file)))
    return cached_data


def ranges_overlap(range1: list, range2: list) -> bool:
    """Check if two time ranges overlap. Assumes ranges are [start, end] in timestamps."""
    return not (
        float(range1[1]) < float(range2[0]) or float(range1[0]) > float(range2[1])
    )


# Example usage:
if __name__ == "__main__":
    data = load_csv(
        exchange="kraken",
        asset="BTC",
        currency="USD",
        filepath="/home/oracle/Downloads/Kraken_Trading_History/XBTUSD.csv",
        begin="2019-07-01",
        end="2020-01-01",
        candle_size=3600,  # 1-hour candles
    )
    print(f"Retrieved data for range: [{data.begin}, {data.end}]")
