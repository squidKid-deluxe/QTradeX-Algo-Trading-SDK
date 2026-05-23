import json
import math
import os
import time
from datetime import datetime

import ccxt
import numpy as np
from qtradex.common.json_ipc import json_ipc
from qtradex.common.utilities import it
from qtradex.core.quant import filter_glitches
from qtradex.public.klines_alphavantage import (
    klines_alphavantage_crypto,
    klines_alphavantage_forex,
    klines_alphavantage_stocks,
)
from qtradex.public.klines_bitshares import klines_bitshares
from qtradex.public.klines_ccxt import BadTimeframeError, klines_ccxt
from qtradex.public.klines_cryptocompare import klines_cryptocompare
from qtradex.public.klines_fdr import klines_fdr
from qtradex.public.klines_synthetic import klines_synthetic
from qtradex.public.klines_yahoo import klines_yahoo
from qtradex.public.utilities import (
    clip_to_time_range,
    implied,
    invert,
    merge_candles,
    quantize_unix,
    reaggregate,
)

DETAIL = False


def parse_date(date_str):
    # Check if the input is a Unix timestamp (integer or float)
    if isinstance(date_str, (int, float)):
        return date_str
    # Try parsing the date in the YY-MM-DD format
    try:
        return time.mktime(datetime.strptime(date_str, "%Y-%m-%d").timetuple())
    except ValueError:
        raise ValueError(
            f"Date must be in YYYY-MM-DD format or a Unix timestamp; got {date_str}"
        )


class Data:
    """
    Gather backtest data.
    """

    def __init__(
        self,
        exchange,
        asset,
        currency,
        begin,
        end=None,
        days=None,
        candle_size=86400,
        pool=None,
        api_key=None,
        intermediary=None,
        placeholder=False,
    ):
        """
        See type(self) for accurate signature.
        """
        # Parse begin and end timestamps as given by user
        if days is not None and end is not None:
            raise ValueError("`days` OR `end` may be given, not both.")

        self.begin = parse_date(begin)
        if end is None and days is not None:
            self.end = int(self.begin - (86400 * days))
        elif days is None and end is not None:
            self.end = parse_date(end)
        else:
            # Default to now
            self.end = int(time.time())

        self.days = (self.end - self.begin) / 86400

        # Add constants to self space
        self.exchange = exchange
        self.asset = asset
        self.currency = currency
        self.pool = pool
        self.candle_size = int(candle_size)
        self.base_size = int(candle_size)
        self.begin = math.ceil(self.begin / candle_size) * candle_size
        self.end = math.ceil(self.end / candle_size) * candle_size
        self.fine_data = None
        self.api_key = api_key

        if self.pool is not None and exchange != "bitshares":
            raise ValueError(
                "Cannot get liquidity pool data for non-bitshares exchange."
            )

        self.raw_candles = {}

        self.intermediary = intermediary
        if not placeholder:
            if intermediary is None:
                self.raw_candles = self.retrieve_and_cache_candles(
                    self.candle_size, self.asset, self.currency
                )
            else:
                if DETAIL:
                    print(f"Using {intermediary} to create implied price...")
                try:
                    self.raw_candles = implied(
                        self.retrieve_and_cache_candles(
                            self.candle_size, self.asset, self.intermediary
                        ),
                        self.retrieve_and_cache_candles(
                            self.candle_size, self.intermediary, self.currency
                        ),
                    )
                except Exception as e:
                    if DETAIL:
                        print(
                            f"Intermediary chain failed: {e}, falling back to direct pair..."
                        )
                    self.raw_candles = self.retrieve_and_cache_candles(
                        self.candle_size, self.asset, self.currency
                    )

            if np.any(self.raw_candles["unix"]):
                self.raw_candles["unix"] = quantize_unix(
                    self.raw_candles["unix"], self.candle_size
                )

                self.begin = np.min(self.raw_candles["unix"])
                self.end = np.max(self.raw_candles["unix"])
            # else:
            #     raise RuntimeError(
            #         f"{self.exchange} does not provide {self.asset}/{self.currency} for this time range."
            #     )

    def __repr__(self):
        """
        <Data object>({candles} candles of data from {exchange}; {begin} to {end}; last price is {last})
        """
        return Data.__repr__.__doc__.strip("\n ").format(
            candles=len(self.raw_candles["close"]),
            exchange=self.exchange,
            begin=time.ctime(self.begin),
            end=time.ctime(self.end),
            last=self.raw_candles["close"][-1],
        )

    def __getitem__(self, index):
        return self.raw_candles[index]

    def update_candles(self, begin, end):
        """
        Re-initialize this class with new start and end.  This method is provided mostly
        for papertrade and live modes where the backend needs to get fresh data.
        """
        self.raw_candles = Data(
            exchange=self.exchange,
            asset=self.asset,
            currency=self.currency,
            begin=begin,
            end=end,
            candle_size=self.candle_size,
            pool=self.pool,
            api_key=self.api_key,
            intermediary=self.intermediary,
        ).raw_candles
        self.begin = begin
        self.end = end

    def keys(self):
        return self.raw_candles.keys()

    def values(self):
        return self.raw_candles.values()

    def items(self):
        return self.raw_candles.items()

    def retrieve_and_cache_candles(self, candle_size, asset, currency):
        """
        Retrieves and caches candlestick data for the specified exchange, asset, and currency
        over a given time range. If the data for the requested time range already exists in cache,
        it will either merge the cached data with new data or use the cache entirely, depending on
        the overlap between the cached and requested time ranges. The method ensures that only the
        relevant data is gathered, merged, and returned, while also maintaining a persistent index
        for future reference.

        Steps:
        1. Check if the data index exists and load it. If not, initialize a new index.
        2. Construct a unique key based on the exchange, asset, and currency.
        3. If data for the given key is not in the cache, gather new data.
        4. If data is cached, determine the overlap with the requested time range and handle merging
           or retrieving the necessary data.
        5. Merge new and cached data if needed, clip the data to the requested time range,
           and update the cache.
        6. Update the data index to reflect the time range of the newly fetched data.

        Side Effects:
        - Writes new or updated data to `data_index.json` and `"{index_key} candles.json"`.
        - The `self.raw_candles` attribute is updated with the relevant data.

        Raises:
        - RuntimeError: If an unexpected condition occurs during the time range checks (should not happen).
        """
        # try to get the index, otherwise initialize it
        try:
            index = json_ipc("data_index.json")
        except FileNotFoundError:
            json_ipc("data_index.json", "{}")
            index = {}
        # try to get the minimum time period cache, otherwise initialize it
        try:
            min_time = json_ipc("min_time.json")
        except FileNotFoundError:
            json_ipc("min_time.json", "{}")
            min_time = {}
        index_key = str((self.exchange, self.pool, candle_size, asset, currency))
        rev_index_key = str((self.exchange, self.pool, candle_size, currency, asset))
        total_time = [self.begin, self.end]
        raw_candles = None
        # if the exchange hasn't been queried before for this pair or its inverse
        if index_key not in index and rev_index_key not in index:
            # gather data
            if DETAIL:
                print(
                    f"gather: {total_time}  use_cache: N/A  @ {candle_size}, {index_key}"
                )
            raw_candles = self.gather_data(
                candle_size, self.begin, self.end, asset, currency
            )
        else:
            inverted = rev_index_key in index
            index_key = rev_index_key if inverted else index_key
            # localize
            time_range = [quantize_unix(i, candle_size) for i in index[index_key]]
            if DETAIL:
                print(time_range)
                print(total_time)
            gather = None
            use_cache = False
            erase_cache = False
            # completely before or after what we need
            if time_range[1] < self.begin or time_range[0] > self.end:
                gather = [[self.begin, self.end]]
                # TODO if we don't erase the cache here it would leave data gaps
                #      but that should be detected and filled, not overwritten
                #      The difficulty is only filling that gap when it is actually
                #      asked for.
                erase_cache = True
                # Remove old cache files and index entries to prevent stale data
                try:
                    os.remove(f"{index_key} candles.json")
                except FileNotFoundError:
                    pass
                index.pop(index_key, None)
                min_time.pop(index_key, None)

            # within the range of what we need
            elif time_range[0] > self.begin and time_range[1] < self.end:
                # Make two queries to get the data "around" what we already have
                gather = [
                    [self.begin, time_range[0] + candle_size],
                    [time_range[1] - candle_size, self.end],
                ]
                use_cache = True  # time_range[:]

            # covers the end of what we need but not the beginning
            elif time_range[0] > self.begin and time_range[1] >= self.end:
                gather = [[self.begin, time_range[0] + candle_size]]
                use_cache = True  # [time_range[0], self.end]

            # covers the beginning of what we need but not the end
            elif time_range[0] <= self.begin and time_range[1] < self.end:
                gather = [[time_range[1] - candle_size, self.end]]
                use_cache = True  # [self.begin, time_range[1]]

            # all of what we need and potentially more
            elif time_range[0] <= self.begin and time_range[1] >= self.end:
                use_cache = True  # [self.begin, self.end]

            else:
                raise RuntimeError(
                    f"THIS SHOULD NOT HAPPEN!  Debug info: {time_range}, {self.begin} {self.end}"
                )

            data = []
            if DETAIL:
                print(
                    f"gather: {gather}  use_cache: {use_cache}  @ {candle_size}, {index_key}"
                )
            # gather up data from the two sources
            if gather is not None:
                for raw_batch in gather:
                    batch = [max(i, min_time.get(index_key, 0)) for i in raw_batch]
                    if batch[0] == batch[1]:
                        if DETAIL:
                            print(
                                f"Cannot fetch {raw_batch}, cache says this exchange does not go this far back"
                            )
                        continue
                    data.append(self.gather_data(candle_size, *batch, asset, currency))
                    if np.any(data[-1]["unix"]):
                        if batch[0] + candle_size < (mindata := min(data[-1]["unix"])):
                            min_time[index_key] = float(
                                max(min_time.get(index_key, 0), mindata)
                            )
            if use_cache:
                try:
                    cached_data = json_ipc(f"{index_key} candles.json")
                    data.append({k: np.array(v) for k, v in cached_data.items()})
                    if inverted:
                        data[-1] = invert(data[-1])
                except FileNotFoundError:
                    # Cache file was deleted or missing, skip using cache
                    pass
                except (json.JSONDecodeError, TypeError) as e:
                    # Cache file is corrupted, skip using cache
                    if DETAIL:
                        print(f"Cache file corrupted: {e}, fetching fresh data...")
                    pass

            candles = dict()
            if len(data) > 1:
                print(f"Merging {len(data)} candlesets into one...")
                # merge the two data sources
                # this will implicitly never happen if erase_cache is True, though
                # an explicit check might be prudent
                candles = merge_candles(data, candle_size)
            else:
                candles = data[0]

            raw_candles = candles
            if not erase_cache:
                total_time = [
                    min(time_range[0], self.begin),
                    max(time_range[1], self.end),
                ]
            else:
                total_time = [self.begin, self.end]

        # write the new cache with all the data

        # if the last candle is incomplete, don't cache it
        crop = -1 if time.time() - self.end < candle_size else None
        if crop is not None and DETAIL:
            print("Cropping last (incomplete) candle from the cache...")
        # convert numpy arrays to lists and crop if required
        cache = {k: v[:crop].tolist() for k, v in raw_candles.items()}
        # find the actual start and end of the cached data
        try:
            total_time = [min(cache["unix"]), max(cache["unix"])]
        except (KeyError, ValueError, TypeError) as e:
            raise TimeoutError(
                f"{self.exchange} does not provide data for this time range."
            ) from e
        # cache it
        json_ipc(f"{index_key} candles.json", json.dumps(cache))

        # clip the return data to the requested amount
        raw_candles = clip_to_time_range(raw_candles, self.begin, self.end)

        json_ipc("min_time.json", json.dumps(min_time))

        # stow the start and end in the index
        if total_time is not None:
            index[index_key] = total_time
            json_ipc("data_index.json", json.dumps(index))

        return raw_candles

    def gather_data(self, candle_size, begin, end, asset, currency, inverted=False):
        """
        Gathers historical candlestick data for a specified asset and currency pair
        from a variety of supported exchanges and APIs. The method checks the
        `self.exchange` attribute to determine the correct data source and fetches
        the data for the provided time range (`begin` to `end`). It supports a wide
        range of exchanges and data providers, including centralized exchanges,
        BitShares, CryptoCompare, Alpha Vantage, and others.

        Args:
            begin (int): The start of the time range for the candlestick data,
                         typically a Unix timestamp.
            end (int): The end of the time range for the candlestick data,
                       typically a Unix timestamp.

        Returns:
            dict: A dictionary containing the raw candlestick data, where the
                  keys may vary depending on the exchange (e.g., "unix", "open",
                  "high", "low", "close", "volume").

        Raises:
            ValueError: If the exchange is not in the list of supported exchanges.

        Notes:
            - The method supports a variety of exchanges including KuCoin, Kraken,
              Bittrex, Poloniex, Bitfinex, Binance, Coinbase, and others.
            - For some exchanges (e.g., `bitshares`, `cryptocompare`, `alphavantage_*`),
              specific API functions are used to fetch the data.
        """
        if int(min(end, time.time())) - int(begin) < candle_size:
            return {
                "unix": np.array([]),
                "open": np.array([]),
                "high": np.array([]),
                "low": np.array([]),
                "close": np.array([]),
                "volume": np.array([]),
            }
        try:
            if DETAIL:
                print(
                    f"gathering data from {self.exchange}, {begin} to {end}, candle size {candle_size}"
                )
            exchange_functions = {
                "bitshares": klines_bitshares,
                "cryptocompare": klines_cryptocompare,
                "alphavantage stocks": klines_alphavantage_stocks,
                "alphavantage forex": klines_alphavantage_forex,
                "alphavantage crypto": klines_alphavantage_crypto,
                "synthetic": klines_synthetic,
                "yahoo": klines_yahoo,
                "finance data reader": klines_fdr,
            }
            if self.exchange in exchange_functions:
                if self.exchange == "synthetic":
                    raw_candles = exchange_functions[self.exchange]()
                else:
                    raw_candles = exchange_functions[self.exchange](
                        asset,
                        currency,
                        begin,
                        end,
                        candle_size,
                        self.api_key
                        if self.exchange.startswith("alphavantage")
                        else self.pool,
                    )
            elif self.exchange in ccxt.exchanges:
                if DETAIL:
                    print("Using CCXT...")
                raw_candles = klines_ccxt(
                    self.exchange,
                    asset,
                    currency,
                    begin,
                    end,
                    candle_size,
                )
            else:
                raise ValueError(f"Invalid exchange {self.exchange}")
            return raw_candles
        except BadTimeframeError as error:
            # if there is a smaller candle size
            if any(candle_size > i for i in error.args[1]):
                # then we need to gather that smaller size and make new candles
                # that are the right size out of it

                # get the biggest size that is smaller than the one we want
                smaller = max([i for i in error.args[1] if candle_size > i])
                if DETAIL:
                    print(
                        f"Exchange does not provide {candle_size} candles! "
                        f" Requesting next smaller candle ({smaller}) and rebucketing..."
                    )
                data = self.retrieve_and_cache_candles(smaller, asset, currency)
                return reaggregate(data, candle_size)
            # but if there isn't, there is no hope
            else:
                raise
        # TODO go through the other klines scripts and have them raise BadSymbol
        #      instead of IndexErrors or KeyErrors
        # except ccxt.base.errors.BadSymbol:
        except Exception as error:
            if inverted:
                raise
            if DETAIL:
                print(
                    it(
                        "yellow",
                        f"Data gathering failed! {error}  Reversing pair and trying again...",
                    )
                )
            # reverse pair and try again
            asset, currency = currency, asset
            return invert(
                self.gather_data(
                    candle_size, begin, end, asset, currency, inverted=True
                )
            )
