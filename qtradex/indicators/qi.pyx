"""
This module provides implementations of various financial technical indicators 
commonly used in trading and financial analysis. The indicators are designed 
to assist traders and analysts in making informed decisions based on price 
movements and trends in financial markets.

These functions are wrapped as *.pxd and compiled in Cython.
"""

from typing import Dict, Tuple

import cython
import numpy as np
import numpy.typing as npt
from numpy import ndarray
from qtradex.common.utilities import truncate
from qtradex.indicators import tulipy_wrapped as ti
from qtradex.indicators.cache_decorator import cache, float_period

DATA_TYPE = np.float64
MA_TYPES = {
    1: ti.dema,
    2: ti.ema,
    3: ti.hma,
    4: ti.kama,
    5: ti.linreg,
    6: ti.sma,
    7: ti.tema,
    8: ti.trima,
    9: ti.tsf,
    10: ti.wilders,
    11: ti.wma,
    12: ti.zlema,
}

Array = npt.NDArray[DATA_TYPE]


def heikin_ashi(
    hlocv: Dict[str, npt.NDArray[DATA_TYPE]]
) -> Dict[str, npt.NDArray[DATA_TYPE]]:
    """
    Calculate Heikin-Ashi candlestick values from the provided OHLCV data.

    Heikin-Ashi is a modified candlestick charting technique that smooths price data
    to better identify trends. It uses average prices to create a more visually
    appealing representation of price movements.

    Parameters:
    - hlocv: A dictionary containing the following NumPy arrays:
        - "open": A NumPy array of opening prices.
        - "high": A NumPy array of high prices.
        - "low": A NumPy array of low prices.
        - "close": A NumPy array of closing prices.
        - "volume": A NumPy array of trading volumes.

    Returns:
    - A dictionary containing the calculated Heikin-Ashi values:
        - "ha_open": A NumPy array of Heikin-Ashi opening prices.
        - "ha_high": A NumPy array of Heikin-Ashi high prices.
        - "ha_low": A NumPy array of Heikin-Ashi low prices.
        - "ha_close": A NumPy array of Heikin-Ashi closing prices.
        - "ha_volume": A NumPy array of trading volumes.
    """

    open_ = hlocv["open"]
    high = hlocv["high"]
    low = hlocv["low"]
    close = hlocv["close"]

    # Initialize arrays for Heikin-Ashi values
    ha_open = np.zeros_like(close)
    ha_high = np.zeros_like(close)
    ha_low = np.zeros_like(close)
    ha_close = np.zeros_like(close)

    # Calculate Heikin-Ashi values
    ha_open[0] = open_[0]  # The first Heikin-Ashi open is the same as the first open
    for i in range(1, len(close)):
        ha_open[i] = (ha_open[i - 1] + close[i - 1]) / 2
        ha_close[i] = (open_[i] + high[i] + low[i] + close[i]) / 4
        ha_high[i] = max(high[i], ha_open[i], ha_close[i])
        ha_low[i] = min(low[i], ha_open[i], ha_close[i])

    return {
        "ha_open": ha_open,
        "ha_high": ha_high,
        "ha_low": ha_low,
        "ha_close": ha_close,
        "ha_volume": hlocv["volume"],  # Volume remains the same
    }


@cache
@float_period(3, 4, 5, 6)
def ichimoku(
    high: npt.NDArray[DATA_TYPE],
    low: npt.NDArray[DATA_TYPE],
    close: npt.NDArray[DATA_TYPE],
    tenkan_period: cython.int,
    kijun_period: cython.int,
    senkou_b_period: cython.int,
    senkou_span: cython.int,
) -> Tuple[
    npt.NDArray[DATA_TYPE],
    npt.NDArray[DATA_TYPE],
    npt.NDArray[DATA_TYPE],
    npt.NDArray[DATA_TYPE],
    npt.NDArray[DATA_TYPE],
]:
    shape = high.shape[0]

    if shape < max(tenkan_period, kijun_period, senkou_b_period, senkou_span):
        raise ValueError(
            "Input arrays must have at least as many elements as the maximum period."
        )

    tenkan_sen = np.zeros(shape, dtype=DATA_TYPE)
    kijun_sen = np.zeros(shape, dtype=DATA_TYPE)
    senkou_span_a = np.zeros(shape, dtype=DATA_TYPE)
    senkou_span_b = np.zeros(shape, dtype=DATA_TYPE)
    chikou_span = np.zeros(shape, dtype=DATA_TYPE)

    # Calculate Tenkan-sen
    for i in range(tenkan_period - 1, shape):
        tenkan_sen[i] = (
            np.max(high[i - tenkan_period + 1 : i + 1])
            + np.min(low[i - tenkan_period + 1 : i + 1])
        ) / 2

    # Calculate Kijun-sen
    for i in range(kijun_period - 1, shape):
        kijun_sen[i] = (
            np.max(high[i - kijun_period + 1 : i + 1])
            + np.min(low[i - kijun_period + 1 : i + 1])
        ) / 2

    # Calculate Senkou Span A and B
    for i in range(senkou_span - 1, shape):
        senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2

    for i in range(senkou_b_period - 1, shape):
        senkou_span_b[i] = (
            np.max(high[i - senkou_b_period + 1 : i + 1])
            + np.min(low[i - senkou_b_period + 1 : i + 1])
        ) / 2

    # Calculate Chikou Span
    if senkou_span < shape:
        chikou_span[senkou_span:] = close[:-senkou_span]

    return (tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span)


def vortex(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    window: cython.int,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate the Vortex Indicator (VI).

    The Vortex Indicator consists of two oscillators that capture positive and negative trend movement.
    A bullish signal triggers when the positive trend indicator crosses above the negative trend indicator.

    Parameters:
    - high: A NumPy array of shape (n,) containing high prices.
    - low: A NumPy array of shape (n,) containing low prices.
    - close: A NumPy array of shape (n,) containing close prices.
    - window: The period for the Vortex Indicator (default is 14).
    - fillna: If True, fill NaN values (default is False).

    Returns:
    - A tuple containing:
      - 'vip': A NumPy array of shape (n,) containing the positive Vortex Indicator.
      - 'vin': A NumPy array of shape (n,) containing the negative Vortex Indicator.
      - 'vid': A NumPy array of shape (n,) containing the difference between +VI and -VI.
    """

    close_shift = np.roll(close, 1)
    close_shift[0] = np.mean(close)  # Fill the first value
    true_range = np.maximum(
        high - low, np.maximum(np.abs(high - close_shift), np.abs(low - close_shift))
    )

    min_periods = window
    trn = np.convolve(true_range, np.ones(window), mode="valid")  # Rolling sum
    vmp = np.abs(high - np.roll(low, 1))
    vmm = np.abs(low - np.roll(high, 1))

    vip = np.convolve(vmp, np.ones(window), mode="valid") / trn
    vin = np.convolve(vmm, np.ones(window), mode="valid") / trn
    vid = vip - vin

    # Handle NaN values
    vip = np.nan_to_num(vip, nan=1.0)
    vin = np.nan_to_num(vin, nan=1.0)
    vid = np.nan_to_num(vid, nan=0.0)

    return vip, vin, vid


@cache
@float_period(1, 2, 3, 4, 5)
def kst(
    close: np.ndarray,
    roc1_period: cython.int,
    roc2_period: cython.int,
    roc3_period: cython.int,
    roc4_period: cython.int,
    kst_smoothing: cython.int,
) -> Tuple[npt.NDArray[DATA_TYPE], npt.NDArray[DATA_TYPE]]:
    """
    Calculate the KST (Know Sure Thing) indicator.

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - roc1_period: The period for the first rate of change (default is 10).
    - roc2_period: The period for the second rate of change (default is 15).
    - roc3_period: The period for the third rate of change (default is 20).
    - roc4_period: The period for the fourth rate of change (default is 30).
    - kst_smoothing: The period for smoothing the KST (default is 9).

    Returns:
    - A tuple containing the calculated KST components:
      - 'kst'
      - 'kst_signal'
    """

    shape = close.shape[0]

    if shape < max(roc1_period, roc2_period, roc3_period, roc4_period, kst_smoothing):
        raise ValueError(
            "Input array must have enough elements for the specified periods."
        )

    # Calculate Rate of Change (ROC) using ti.roc
    roc1 = ti.roc(close, roc1_period)
    roc2 = ti.roc(close, roc2_period)
    roc3 = ti.roc(close, roc3_period)
    roc4 = ti.roc(close, roc4_period)

    roc1, roc2, roc3, roc4 = truncate(roc1, roc2, roc3, roc4)

    # Calculate KST
    _kst = roc1 * 1 + roc2 * 2 + roc3 * 3 + roc4 * 4
    kst_signal = np.zeros_like(_kst, dtype=np.float64)

    # Smooth the KST using a rolling mean
    for i in range(_kst.shape[0]):
        if i >= kst_smoothing:
            kst_signal[i] = np.mean(_kst[i - kst_smoothing + 1 : i + 1])

    return (
        _kst,
        kst_signal,
    )


@cache
@float_period(1, 2)
def frama(
    close: np.ndarray, period: cython.int, fractal_period: cython.int
) -> npt.NDArray[DATA_TYPE]:
    """
    Calculate the Fractal Adaptive Moving Average (FRAMA).

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - period: The period for calculating the FRAMA (default is 14).
    - fractal_period: The period for calculating the fractal dimension (default is 2).

    Returns:
    - A numpy array containing the calculated FRAMA
    """

    shape = close.shape[0]

    if shape < period:
        raise ValueError(
            "Input array must have enough elements for the specified period."
        )

    _frama = np.zeros(shape, dtype=np.float64)
    fractal_dim = np.zeros(shape, dtype=np.float64)

    # Calculate FRAMA
    for i in range(period - 1, shape):
        sum_high = np.sum(close[i - period + 1 : i + 1])
        sum_low = np.sum(close[i - period + 1 : i + 1])
        sum_close = np.sum(close[i - period + 1 : i + 1])
        _frama[i] = (sum_high + sum_low + sum_close) / (3 * period)

        # Calculate fractal dimension
        if i >= fractal_period - 1:
            fractal_sum = np.sum(close[i - fractal_period + 1 : i + 1])
            fractal_dim[i] = fractal_sum / fractal_period

        # Adjust FRAMA based on fractal dimension
        if fractal_dim[i] > 0:
            _frama[i] += (fractal_dim[i] - _frama[i]) * (1 / fractal_period)

    return _frama


# FIXME
@cache
def zigzag(
    close: npt.NDArray[DATA_TYPE], deviation: float
) -> Tuple[npt.NDArray[DATA_TYPE], npt.NDArray[DATA_TYPE]]:
    """
    Calculate the Zig Zag Indicator.

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - deviation: The percentage change required to identify a reversal (default is 5.0).

    Returns:
    - A tuple containing:
      - 'steps': a step function at each zig zag point
      - 'zigzag': interpolated between steps
    """
    shape = close.shape[0]

    if shape == 0:
        return np.full(shape, np.nan, dtype=DATA_TYPE), np.full(
            shape, np.nan, dtype=DATA_TYPE
        )  # Return early if the input array is empty

    steps = np.full(shape, np.nan, dtype=DATA_TYPE)
    last_extreme = close[0]
    last_direction = 0  # 1 for up, -1 for down
    steps[0] = last_extreme  # Set the first value of steps

    for i in range(1, shape):
        change = (close[i] - last_extreme) / last_extreme * 100

        if change > deviation and last_direction != 1:
            steps[i] = close[i]
            last_extreme = close[i]
            last_direction = 1
        elif change < -deviation and last_direction != -1:
            steps[i] = close[i]
            last_extreme = close[i]
            last_direction = -1
        else:
            steps[i] = steps[i - 1]

    zigzag = np.full(shape, np.nan, dtype=DATA_TYPE)
    valid_indices = np.where(~np.isnan(steps))[0]

    if valid_indices.size > 0:
        for i in range(len(valid_indices) - 1):
            start_idx = valid_indices[i]
            end_idx = valid_indices[i + 1]
            zigzag[start_idx : end_idx + 1] = np.linspace(
                steps[start_idx], steps[end_idx], end_idx - start_idx + 1
            )
        last_valid_index = valid_indices[-1]
        zigzag[last_valid_index:] = steps[
            last_valid_index
        ]  # Fill remaining values with the last valid step

    return zigzag, steps


@cache
@float_period(3, 4)
def ravi(
    high: npt.NDArray[DATA_TYPE],
    low: npt.NDArray[DATA_TYPE],
    close: npt.NDArray[DATA_TYPE],
    short_period: cython.int,
    long_period: cython.int,
) -> npt.NDArray[DATA_TYPE]:
    """
    Calculate the Range Action Verification Index (RAVI).

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - high: A NumPy array of shape (n,) containing high prices.
    - low: A NumPy array of shape (n,) containing low prices.
    - short_period: The short period for RAVI calculation (default is 14).
    - long_period: The long period for RAVI calculation (default is 30).

    Returns:
    - A tuple containing the calculated RAVI values:
      - 'ravi': The RAVI values
    """
    shape = close.shape[0]
    if shape == 0:
        return np.array([])  # Handle empty input

    _ravi = np.zeros(shape, dtype=DATA_TYPE)
    short_avg_range = np.zeros(shape, dtype=DATA_TYPE)
    long_avg_range = np.zeros(shape, dtype=DATA_TYPE)

    # Calculate average ranges using vectorized operations
    for i in range(shape):
        if i >= short_period - 1:
            short_avg_range[i] = np.mean(high[i - short_period + 1 : i + 1]) - np.mean(
                low[i - short_period + 1 : i + 1]
            )
        if i >= long_period - 1:
            long_avg_range[i] = np.mean(high[i - long_period + 1 : i + 1]) - np.mean(
                low[i - long_period + 1 : i + 1]
            )

    # Calculate RAVI
    valid_long_avg_range = long_avg_range[
        long_period - 1 :
    ]  # Only consider valid long averages
    _ravi[long_period - 1 :] = (
        (short_avg_range[long_period - 1 :] - valid_long_avg_range)
        / valid_long_avg_range
        * 100
    )
    _ravi[long_avg_range == 0] = 0.0  # Handle division by zero

    return _ravi


@cache
@float_period(1)
def aema(
    close: npt.NDArray[DATA_TYPE], period: cython.int, alpha: float = 0.1
) -> npt.NDArray[DATA_TYPE]:
    """
    Calculate the Adaptive Exponential Moving Average (AEMA).

    The AEMA adjusts the smoothing factor based on the volatility of the price
    changes, allowing it to respond more quickly to price movements.

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - period: The period for calculating the AEMA (default is 14).
    - alpha: The base smoothing factor (default is 0.1).

    Returns:
    - A tuple containing the calculated AEMA values:
      - 'aema': The AEMA values as a NumPy array.
    """
    shape = close.shape[0]
    if shape == 0:
        return (np.array([]),)  # Handle empty input

    _aema = np.zeros(shape, dtype=DATA_TYPE)

    # Initialize the first AEMA value using the average of the first 'period' values
    if shape >= period:
        _aema[0] = np.mean(
            close[:period]
        )  # Start with the average of the first 'period' closes
    else:
        _aema[0] = close[0]  # Fallback for very short arrays

    # Calculate AEMA
    for i in range(1, shape):
        volatility = abs(close[i] - close[i - 1])
        adjusted_alpha = alpha / (1 + volatility)  # Adjust alpha based on volatility
        _aema[i] = (adjusted_alpha * close[i]) + ((1 - adjusted_alpha) * _aema[i - 1])

    return _aema


@cache
@float_period(1, 2, 3)
def typed_macd(
    close: npt.NDArray[DATA_TYPE],
    short_period: cython.int,
    long_period: cython.int,
    signal_period: cython.int,
    ma_type: cython.int,  # New argument for moving average type
) -> Tuple[npt.NDArray[DATA_TYPE], npt.NDArray[DATA_TYPE], npt.NDArray[DATA_TYPE]]:
    """
    Calculate the Moving Average Convergence Divergence (MACD) using specified moving average type.

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - short_period: The short period for the moving average (default is 12).
    - long_period: The long period for the moving average (default is 26).
    - signal_period: The period for the signal line (default is 9).
    - ma_type: The type of moving average to use ('ema', 'sma', etc.).

    Returns:
    - A tuple containing the calculated MACD values:
      - 'macd': The MACD values
      - 'signal_line': The signal line values
      - 'histogram': The histogram values (MACD - signal line)
    """

    if close.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])  # Handle empty input

    # Select the moving average function based on the ma_type argument
    if ma_type not in MA_TYPES:
        raise ValueError(f"Unsupported moving average type: {ma_type}")

    function = MA_TYPES[ma_type]

    # Calculate the short and long moving averages using the selected function
    ma_short = function(close, short_period)
    ma_long = function(close, long_period)

    ma_short, ma_long = truncate(ma_short, ma_long)

    # Calculate MACD
    macd = ma_short - ma_long

    # Calculate the signal line (moving average of MACD)
    signal_line = function(macd, signal_period)

    macd, signal_line = truncate(macd, signal_line)

    # Calculate the histogram
    histogram = macd - signal_line

    return (
        macd,
        signal_line,
        histogram,
    )


@cache
@float_period(1, 3)
def typed_bbands(
    close: npt.NDArray[DATA_TYPE],
    ma_period: cython.int,
    ma_type: cython.int,  # New argument for moving average type
    std_period: cython.int,  # Seperate standard deviation period
    deviations: float,
) -> Tuple[npt.NDArray[DATA_TYPE], npt.NDArray[DATA_TYPE], npt.NDArray[DATA_TYPE]]:
    """
    Calculate the Bollinger Bands using specified moving average type.

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - period: The period for the moving average (default is 20).
    - deviations: The number of standard deviations to use for the bands.
    - ma_type: The type of moving average to use ('ema', 'sma', etc.).

    Returns:
    - A tuple containing the calculated Bollinger Bands:
      - 'middle_band': The middle band (moving average)
      - 'upper_band': The upper band (middle band + deviations * std deviation)
      - 'lower_band': The lower band (middle band - deviations * std deviation)
    """

    if close.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])  # Handle empty input

    # Select the moving average function based on the ma_type argument
    if ma_type not in MA_TYPES:
        raise ValueError(f"Unsupported moving average type: {ma_type}")

    function = MA_TYPES[ma_type]

    # Calculate the middle band (moving average)
    middle_band = function(close, ma_period)

    # Calculate the rolling standard deviation
    rolling_std = np.std(
        close[-std_period:], ddof=1
    )  # Use the last 'period' values for std deviation

    # Calculate the upper and lower bands
    upper_band = middle_band + (deviations * rolling_std)
    lower_band = middle_band - (deviations * rolling_std)

    return upper_band, middle_band, lower_band


@cache
@float_period(1, 2)
def tsi(
    close: np.ndarray, long_period: cython.int, short_period: cython.int
) -> npt.NDArray[DATA_TYPE]:
    """
    Calculate the Trend Strength Indicator (TSI).

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - long_period: The long period for TSI calculation (default is 25).
    - short_period: The short period for TSI calculation (default is 13).

    Returns:
    - A tuple containing the calculated TSI values:
      - 'tsi': The TSI values
    """
    shapeshape = close.shape[0]
    _tsi = np.zeros(shapeshape, dtype=np.float64)
    price_change = np.diff(close, prepend=close[0])

    smoothed_price_change = np.zeros(shapeshape, dtype=np.float64)
    smoothed_abs_price_change = np.zeros(shapeshape, dtype=np.float64)

    # Calculate smoothed price change (double smoothing)
    for i in range(shapeshape):
        if i == 0:
            smoothed_price_change[i] = price_change[i]
            smoothed_abs_price_change[i] = abs(price_change[i])
        else:
            if i < short_period:
                smoothed_price_change[i] = (
                    smoothed_price_change[i - 1] * (i - 1) + price_change[i]
                ) / i
                smoothed_abs_price_change[i] = (
                    smoothed_abs_price_change[i - 1] * (i - 1) + abs(price_change[i])
                ) / i
            else:
                smoothed_price_change[i] = (
                    smoothed_price_change[i - 1] * (short_period - 1) + price_change[i]
                ) / short_period
                smoothed_abs_price_change[i] = (
                    smoothed_abs_price_change[i - 1] * (short_period - 1)
                    + abs(price_change[i])
                ) / short_period

    # Calculate TSI
    _tsi[long_period:] = (
        smoothed_price_change[long_period:] / smoothed_abs_price_change[long_period:]
    ) * 100

    return _tsi


@cache
@float_period(3, 4)
def smi(
    close: npt.NDArray[DATA_TYPE],
    high: npt.NDArray[DATA_TYPE],
    low: npt.NDArray[DATA_TYPE],
    k_period: cython.int,
    d_period: cython.int,
) -> Tuple[npt.NDArray[DATA_TYPE], npt.NDArray[DATA_TYPE]]:
    """
    Calculate the Stochastic Momentum Index (SMI).

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - high: A NumPy array of shape (n,) containing high prices.
    - low: A NumPy array of shape (n,) containing low prices.
    - k_period: The period for the SMI calculation (default is 14).
    - d_period: The period for the signal line (default is 3).

    Returns:
    - A tuple containing:
      - 'smi': The Stochastic Momentum Index values
      - 'smi_signal': The smoothed SMI values
    """
    shape = close.shape[0]
    if shape == 0:
        return (np.array([]), np.array([]))  # Handle empty input

    _smi = np.zeros(shape, dtype=DATA_TYPE)
    smi_signal = np.zeros(shape, dtype=DATA_TYPE)

    # Calculate SMI values
    for i in range(k_period - 1, shape):
        highest_high = np.max(high[i - k_period + 1 : i + 1])
        lowest_low = np.min(low[i - k_period + 1 : i + 1])
        if highest_high != lowest_low:
            _smi[i] = (
                (close[i] - (highest_high + lowest_low) / 2)
                / ((highest_high - lowest_low) / 2)
            ) * 100
        else:
            _smi[i] = 0.0

    # Smooth the SMI values
    for i in range(d_period - 1, shape):
        smi_signal[i] = np.mean(_smi[i - d_period + 1 : i + 1])

    return _smi, smi_signal


@cache
@float_period(3)
def eri(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ma_period: cython.int,
    ma_type: cython.int,
) -> Tuple[npt.NDArray[DATA_TYPE], npt.NDArray[DATA_TYPE]]:
    """
    Calculate the Elder Ray Index, which uses bull and bear power to gauge market strength
    and potential reversals.

    The Elder Ray Index consists of two components:
    - Bull Power: The difference between the highest price of the current period and the
      exponential moving average (EMA) of the closing prices.
    - Bear Power: The difference between the lowest price of the current period and the
      EMA of the closing prices.

    Parameters:
    ----------
    close :  A 1D array of closing prices.
    high : A 1D array of high prices.
    low : A 1D array of low prices.
    ema_period : The period for calculating the Exponential Moving Average

    Returns:
    -------
    tuple
        A tuple containing two 1D arrays:
        - bull_power : ndarray
            A 1D array of Bull Power values.
        - bear_power : ndarray
            A 1D array of Bear Power values.
    """

    # Calculate moving average using the provided ti function
    function = MA_TYPES[ma_type]
    ma_values = function(close, ma_period)
    high, low, ma_values = truncate(high, low, ma_values)
    shape = ma_values.shape[0]
    bull_power = np.zeros(shape, dtype=np.float64)
    bear_power = np.zeros(shape, dtype=np.float64)

    for i in range(shape):
        bull_power[i] = high[i] - ma_values[i]
        bear_power[i] = low[i] - ma_values[i]

    return (
        bull_power,
        bear_power,
    )


@cache
@float_period(3)
def supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    multiplier_top: float,
    multiplier_bottom: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Supertrend indicator.

    Parameters:
    - high (array-like): Array of high prices.
    - low (array-like): Array of low prices.
    - close (array-like): Array of close prices.
    - period (int): Period for ATR calculation (default: 14).
    - multiplier (float): Multiplier for the bands (default: 3).
    """
    trailing_high = np.empty(high.shape[0] - period)
    trailing_low = np.empty(low.shape[0] - period)
    for idx in range(len(high) - period):
        trailing_high[idx] = np.max(high[idx : idx + period])
        trailing_low[idx] = np.min(low[idx : idx + period])

    # Calculate the ATR using Tulipy
    atr = ti.atr(high, low, close, period)

    # Calculate the average of high and low
    hla = (trailing_high + trailing_low) / 2
    hla, atr = truncate(hla, atr)

    # Initialize basic upper and lower bands
    upperband = hla + (multiplier_top * atr)
    lowerband = hla - (multiplier_bottom * atr)
    close, _ = truncate(close, atr)

    supertrend = []
    toggle = close[0] > lowerband[0]
    for u, l, c in zip(upperband, lowerband, close):
        if toggle and c < l:
            toggle = False
        elif not toggle and u < c:
            toggle = True
        if toggle:
            supertrend.append(l)
        else:
            supertrend.append(u)
    return supertrend, upperband, lowerband


@cache
@float_period(1)
def arsi(
    close: npt.NDArray[DATA_TYPE], length: cython.int = 14
) -> Tuple[npt.NDArray[DATA_TYPE]]:
    """
    Calculate the Adaptive Relative Strength Index (ARSI).

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - length: The period for calculating the traditional RSI (default is 14).
    - highlight_movements: Whether to highlight movements (default is True).

    Returns:
    - A NumPy array containing the calculated Adaptive RSI values.
    """

    # Calculate the RSI using qx.ti.rsi
    rsi = ti.rsi(close, length)

    # Calculate the Adaptive RSI
    rsi, close = truncate(rsi, close)
    shape = close.shape[0]
    arsi = np.zeros(shape, dtype=DATA_TYPE)

    for i in range(shape):
        alpha = 2 * abs(rsi[i] / 100 - 0.5)
        if i == 0:
            arsi[i] = close[i]  # Initialize the first value
        else:
            arsi[i] = alpha * close[i] + (1 - alpha) * arsi[i - 1]

    return (arsi,)


@cache
@float_period(3, 4)
def keltner(
    high: npt.NDArray[DATA_TYPE],
    low: npt.NDArray[DATA_TYPE],
    close: npt.NDArray[DATA_TYPE],
    atr_period: cython.int,
    ma_period: cython.int,
    ma_type: cython.int,
    multiplier: float = 1.5,
) -> Tuple[npt.NDArray[DATA_TYPE], npt.NDArray[DATA_TYPE], npt.NDArray[DATA_TYPE]]:
    """
    Calculate the Keltner Channels.

    Parameters:
    - high: A NumPy array of high prices.
    - low: A NumPy array of low prices.
    - close: A NumPy array of close prices.
    - atr_period: The period for calculating the ATR (default is 20).
    - ema_period: The period for calculating the EMA (default is 20).
    - multiplier: The multiplier for the ATR to set the channel width (default is 1.5).

    Returns:
    - A tuple containing the Keltner Channel values:
      - upper_band: The upper Keltner Channel values
      - middle_band: The middle Keltner Channel (EMA) values
      - lower_band: The lower Keltner Channel values
    """
    # Calculate the Average True Range (ATR)
    atr_values = ti.atr(high, low, close, atr_period)

    # Calculate the Exponential Moving Average (EMA)
    function = MA_TYPES[ma_type]
    middle_band = function(close, ma_period)

    atr_values, middle_band = truncate(atr_values, middle_band)

    # Initialize arrays for the upper and lower bands
    upper_band = np.full(middle_band.shape, np.nan, dtype=close.dtype)
    lower_band = np.full(middle_band.shape, np.nan, dtype=close.dtype)

    # Calculate the upper and lower bands
    for i, middle in enumerate(middle_band):
        if np.isnan(atr_values[i]) or np.isnan(middle):
            continue
        upper_band[i] = middle + (multiplier * atr_values[i])
        lower_band[i] = middle - (multiplier * atr_values[i])

    return upper_band, middle_band, lower_band


@cache
@float_period(2)
def donchian(
    high: npt.NDArray[DATA_TYPE], low: npt.NDArray[DATA_TYPE], period: cython.int
) -> Tuple[npt.NDArray[DATA_TYPE], npt.NDArray[DATA_TYPE], npt.NDArray[DATA_TYPE]]:
    """
    Calculate the Donchian Channels.

    Parameters:
    - high: A NumPy array of high prices.
    - low: A NumPy array of low prices.
    - period: The period for calculating the Donchian Channels (default is 20).

    Returns:
    - A tuple containing the Donchian Channel values:
      - upper_band: The upper Donchian Channel values
      - lower_band: The lower Donchian Channel values
      - middle_band: The middle Donchian Channel values (average of upper and lower bands)
    """
    shape = high.shape[0]
    upper_band = np.full(shape, np.nan, dtype=high.dtype)
    lower_band = np.full(shape, np.nan, dtype=low.dtype)
    middle_band = np.full(shape, np.nan, dtype=high.dtype)

    for i in range(period - 1, shape):
        upper_band[i] = np.max(high[i - period + 1 : i + 1])
        lower_band[i] = np.min(low[i - period + 1 : i + 1])
        middle_band[i] = (upper_band[i] + lower_band[i]) / 2

    return upper_band, middle_band, lower_band


@cache
def kagi(
    close: npt.NDArray[DATA_TYPE], reversal_percent: float
) -> Tuple[npt.NDArray[DATA_TYPE]]:
    """
    Calculate the Kagi indicator based on price movements and a specified reversal percentage.

    Parameters:
    prices (np.ndarray): Array of price data.
    reversal_percent (float): Percentage change required for a reversal (e.g., 2 for 2%).

    Returns:
    np.ndarray: Kagi indicator values.
    """
    # Convert percentage to a fraction
    reversal_fraction = reversal_percent / 100

    # Initialize variables
    kagi = []
    direction = None  # None, 'up', or 'down'
    last_price = None

    for price in close:
        if last_price is None:
            # Initialize the first price
            kagi.append(price)
            last_price = price
            continue

        # Calculate the reversal amount based on the last price
        reversal_amount_up = last_price * (
            1 + reversal_fraction
        )  # Price must rise by reversal_fraction
        reversal_amount_down = last_price * (
            1 - reversal_fraction
        )  # Price must fall by reversal_fraction

        # Determine the direction of the Kagi line
        if (
            direction is None
            or (direction == "up" and price < reversal_amount_down)
            or (direction == "down" and price > reversal_amount_up)
        ):
            # Reverse the direction
            if direction == "up":
                direction = "down"
            else:
                direction = "up"
            kagi.append(price)  # Add the current price as the new Kagi point
            last_price = price  # Update the last price
        elif direction == "up" and price > last_price:
            # Continue in the upward direction
            kagi.append(price)
            last_price = price
        elif direction == "down" and price < last_price:
            # Continue in the downward direction
            kagi.append(price)
            last_price = price
        else:
            # No change in direction, keep the last Kagi point
            kagi.append(kagi[-1])

    print("kagi", kagi, "\n", np.array(kagi))
    return (np.array(kagi),)


@cache
def renko(
    close: npt.NDArray[DATA_TYPE], brick_percent: float
) -> Tuple[npt.NDArray[DATA_TYPE]]:
    """
    Calculate the Renko Chart values using a percentage-based brick size.

    Parameters:
    - close: A NumPy array of price data.
    - brick_percent: The size of each Renko brick as a percentage of the current price.

    Returns:
    - A NumPy array containing the Renko Chart values.
    """
    if len(close) == 0:
        return np.array([])

    renko_values = []
    last_brick = None

    for price in close:
        if last_brick is None:
            last_brick = price

        # Calculate the brick size based on the current price
        brick_size = price * (brick_percent / 100)

        while price >= last_brick + brick_size:
            last_brick += brick_size
            renko_values.append(last_brick)

        while price <= last_brick - brick_size:
            last_brick -= brick_size
            renko_values.append(last_brick)

    return (np.array(renko_values),)


@cache
def tick_indicator(close: npt.NDArray[DATA_TYPE]) -> npt.NDArray[DATA_TYPE]:
    """
    Calculate the TICK indicator based on close.

    Parameters:
    - close: A NumPy array of closing prices.

    Returns:
    - A NumPy array containing the TICK values.
    """
    if len(close) < 2:
        return np.array([])

    # Calculate upticks and downticks
    upticks = np.where(close[1:] > close[:-1], 1, 0)
    downticks = np.where(close[1:] < close[:-1], 1, 0)

    tick_values = upticks - downticks
    return tick_values


@cache
def trin_indicator(
    close: npt.NDArray[DATA_TYPE], volume: npt.NDArray[DATA_TYPE]
) -> npt.NDArray[DATA_TYPE]:
    if len(close) != len(volume):
        raise ValueError("Close prices and volumes must have the same length.")

    # Determine advancing and declining issues
    advancing = close[1:] > close[:-1]
    declining = close[1:] < close[:-1]

    # Create arrays for advancing and declining issues and volumes
    advancing_issues = np.zeros(len(close) - 1)
    declining_issues = np.zeros(len(close) - 1)
    advancing_volume = np.zeros(len(close) - 1)
    declining_volume = np.zeros(len(close) - 1)

    advancing_issues[advancing] = 1
    declining_issues[declining] = 1
    advancing_volume[advancing] = volume[1:][advancing]
    declining_volume[declining] = volume[1:][declining]

    # Calculate cumulative sums and add 1 to avoid division by zero
    adv_issues_cumsum = np.cumsum(advancing_issues) + 1
    decl_issues_cumsum = np.cumsum(declining_issues) + 1
    adv_volume_cumsum = np.cumsum(advancing_volume) + 1
    decl_volume_cumsum = np.cumsum(declining_volume) + 1

    # Calculate TRIN
    trin = (adv_issues_cumsum / decl_issues_cumsum) / (
        adv_volume_cumsum / decl_volume_cumsum
    )

    # Prepend a value for the first element (since we calculated from the second element)
    trin = np.insert(trin, 0, 1)  # You can choose a suitable value for the first TRIN

    return trin


@cache
def market_profile(
    close: npt.NDArray[DATA_TYPE], volume: npt.NDArray[DATA_TYPE], bin_size: float
) -> Tuple[npt.NDArray[DATA_TYPE], npt.NDArray[DATA_TYPE]]:
    """
    Calculate the Market Profile values.

    Parameters:
    - close: A NumPy array of price data.
    - volume: A NumPy array of volume data corresponding to the close.
    - bin_size: The size of each price bin for the Market Profile.

    Returns:
    - A tuple containing:
      - price_bins: A NumPy array of price bins.
      - volume_profile: A NumPy array of volume for each price bin.
    """
    if len(close) == 0 or len(volume) == 0:
        return np.array([]), np.array([])

    min_price = np.floor(np.min(close) / bin_size) * bin_size
    max_price = np.ceil(np.max(close) / bin_size) * bin_size
    num_bins = int((max_price - min_price) / bin_size) + 1

    price_bins = np.arange(min_price, max_price, bin_size)
    volume_profile = np.zeros(num_bins)

    for i in range(len(close)):
        bin_index = int((close[i] - min_price) / bin_size)
        if 0 <= bin_index < num_bins:
            volume_profile[bin_index] += volume[i]

    return price_bins, volume_profile


@cache
@float_period(1)
def price_action(
    close: npt.NDArray[DATA_TYPE], lookback: cython.int, threshold: float = 0.01
) -> Tuple[npt.NDArray[DATA_TYPE], npt.NDArray[DATA_TYPE]]:
    """
    Identify support and resistance levels based on price action.

    Parameters:
    - close: A NumPy array of price data.
    - lookback: The number of periods to look back for identifying levels.
    - threshold: The percentage threshold to consider a level significant.

    Returns:
    - A tuple containing:
      - support_levels: A NumPy array of identified support levels.
      - resistance_levels: A NumPy array of identified resistance levels.
    """
    if len(close) == 0:
        return np.array([]), np.array([])

    support_levels = []
    resistance_levels = []

    for i in range(lookback, len(close)):
        current_price = close[i]
        past_prices = close[i - lookback : i]

        # Identify support level
        if current_price < np.min(past_prices) * (1 + threshold):
            support_levels.append(current_price)

        # Identify resistance level
        if current_price > np.max(past_prices) * (1 - threshold):
            resistance_levels.append(current_price)

    return np.array(support_levels), np.array(resistance_levels)


@cache
@float_period(1)
def holt_winters_des(
    x: npt.NDArray[np.float64],
    span: int,
    beta: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    The Holt-Winters second order method (double exponential smoothing) incorporates the estimated
    trend into the smoothed data, using a trend term that keeps track of the slope of the original signal.

    Parameters:
    - x: A NumPy array of shape (n,) containing the input data.
    - span: The number of data points taken for calculation.
    - beta: The trend smoothing factor, 0 < beta < 1.

    Returns:
    - A tuple containing:
      - smoothed_series: A NumPy array of shape (n,) containing the smoothed values.
      - smoothed_trend: A NumPy array of shape (n,) containing the smoothed trend values.
    """

    if span < 0:
        raise ValueError("span must be >= 0")
    if beta <= 0 or beta >= 1:
        raise ValueError("beta must be in the range (0, 1)")

    x = np.reshape(x, (x.shape[0], -1))
    alpha = 2.0 / (1 + span)
    r_alpha = 1 - alpha
    r_beta = 1 - beta
    s = np.zeros_like(x)
    b = np.zeros_like(x)
    s[0, :] = x[0, :]

    for i in range(1, x.shape[0]):
        s[i, :] = alpha * x[i, :] + r_alpha * (s[i - 1, :] + b[i - 1, :])
        b[i, :] = beta * (s[i, :] - s[i - 1, :]) + r_beta * b[i - 1, :]

    return s, b


@cache
@float_period(1)
def ulcer_index(
    close: npt.NDArray[np.float64],
    window: cython.int,
) -> npt.NDArray[np.float64]:
    """
    Calculate the Ulcer Index (UI).

    The Ulcer Index measures the depth and duration of drawdowns in price.

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - window: The period for the Ulcer Index (default is 14).
    - fillna: If True, fill NaN values (default is False).

    Returns:
    - A NumPy array of shape (n,) containing the Ulcer Index values.
    """

    # Calculate the maximum close price over the window
    ui_max = np.maximum.accumulate(close)  # Cumulative maximum
    r_i = 100 * (close - ui_max) / ui_max

    # Calculate the Ulcer Index using a rolling window
    ulcer_idx = np.zeros_like(close, dtype=np.float64)

    for i in range(window, len(close)):
        # Calculate the standard deviation of the drawdowns over the window
        drawdowns = r_i[i - window + 1 : i + 1]
        ulcer_idx[i] = np.sqrt(np.sum(drawdowns**2) / window)

    # Handle NaN values
    ulcer_idx = np.nan_to_num(ulcer_idx, nan=0.0)

    return ulcer_idx


@cache
@float_period(1)
def trix(
    close: npt.NDArray[np.float64],
    window: cython.int,
) -> npt.NDArray[np.float64]:
    """
    Calculate the Trix (TRIX) indicator.

    The Trix indicator shows the percent rate of change of a triple exponentially smoothed moving average.

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - window: The period for the Trix indicator (default is 15).
    - fillna: If True, fill NaN values (default is False).

    Returns:
    - A NumPy array of shape (n,) containing the Trix values.
    """

    # Calculate the triple EMA
    ema1 = ti.ema(close, window)
    ema2 = ti.ema(ema1, window)
    ema3 = ti.ema(ema2, window)

    # Calculate TRIX
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # Set the first value to 0 or handle it as needed
    trix *= 100  # Convert to percentage

    # Handle NaN values
    trix = np.nan_to_num(trix, nan=0.0)

    return trix


@cache
@float_period(1, 2, 3)
def earsi(
    close: npt.NDArray[DATA_TYPE],
    auto_min: cython.int,
    auto_max: cython.int,
    auto_avg: cython.int,
) -> Tuple[npt.NDArray[DATA_TYPE]]:
    """
    Calculate the Ehlers Adaptive Relative Strength Index (EARSI).

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - auto_min: The minimum length for adaptive calculation (default is 10).
    - auto_max: The maximum length for adaptive calculation (default is 48).
    - auto_avg: The average length for adaptive calculation (default is 3).

    Returns:
    - A NumPy array containing the calculated Adaptive RSI values.
    """
    # Ensure auto_avg is within the bounds of auto_min and auto_max
    auto_avg = max(auto_min, min(auto_avg, auto_max))

    shape = close.shape[0]
    rsi = np.zeros(shape, dtype=DATA_TYPE)
    adaptive_rsi = np.zeros(shape, dtype=DATA_TYPE)

    # Calculate gains and losses
    changes = np.diff(close, prepend=close[0])
    gain = np.maximum(changes, 0)
    loss = -np.minimum(changes, 0)

    # Calculate average gains and losses
    avg_gain = np.zeros(shape, dtype=DATA_TYPE)
    avg_loss = np.zeros(shape, dtype=DATA_TYPE)

    avg_gain[auto_min - 1] = np.mean(gain[:auto_min])
    avg_loss[auto_min - 1] = np.mean(loss[:auto_min])

    for i in range(auto_min, shape):
        avg_gain[i] = (avg_gain[i - 1] * (auto_min - 1) + gain[i]) / auto_min
        avg_loss[i] = (avg_loss[i - 1] * (auto_min - 1) + loss[i]) / auto_min

    strength = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi[auto_min:] = 100 - (100 / (1 + strength[auto_min:]))

    def high_pass_filter(
        src: npt.NDArray[DATA_TYPE], cutoff: cython.int
    ) -> npt.NDArray[DATA_TYPE]:
        """
        Apply a high-pass filter to the input signal.

        Parameters:
        - src: A NumPy array of shape (n,) containing the input signal.
        - cutoff: The cutoff period for the high-pass filter.

        Returns:
        - A NumPy array containing the filtered signal.
        """
        filtered = np.zeros_like(src)
        for i in range(len(src)):
            if i < cutoff:
                filtered[i] = src[i]  # No filtering for the initial values
            else:
                filtered[i] = src[i] - np.mean(
                    src[i - cutoff : i]
                )  # Subtract the moving average

        return filtered

    # Calculate adaptive period
    def adaptive_period(
        src: npt.NDArray[DATA_TYPE],
        min_len: cython.int,
        max_len: cython.int,
        ave_len: cython.int,
    ) -> float:
        filtered = high_pass_filter(src, max_len)
        corr = np.zeros(max_len * 2)
        cos_part = np.zeros(max_len * 2)
        sin_part = np.zeros(max_len * 2)
        sq_sum = np.zeros(max_len * 2)

        for lag in range(max_len):
            m = ave_len if ave_len != 0 else lag
            Sx = Sy = Sxx = Syy = Sxy = 0.0
            for i in range(m):
                x = filtered[i] if i < len(filtered) else 0
                y = filtered[lag + i] if lag + i < len(filtered) else 0
                Sx += x
                Sy += y
                Sxx += x * x
                Sxy += x * y
                Syy += y * y
            if (m * Sxx - Sx * Sx) * (m * Syy - Sy * Sy) > 0:
                corr[lag] = (m * Sxy - Sx * Sy) / np.sqrt(
                    (m * Sxx - Sx * Sx) * (m * Syy - Sy * Sy)
                )

        for period in range(min_len, max_len + 1):
            for n in range(ave_len, max_len + 1):
                cos_part[period] += corr[n] * np.cos(2 * np.pi * n / period)
                sin_part[period] += corr[n] * np.sin(2 * np.pi * n / period)
            sq_sum[period] = cos_part[period] ** 2 + sin_part[period] ** 2

        max_power = np.max(sq_sum[min_len : max_len + 1])
        pwr = np.where(max_power != 0, sq_sum / max_power, 0)

        spx = sp = 0.0
        for period in range(min_len, max_len + 1):
            if pwr[period] >= 0.5:
                spx += period * pwr[period]
                sp += pwr[period]

        dominant_cycle = spx / sp if sp != 0 else 0
        return np.clip(dominant_cycle, min_len, max_len)

    # Calculate the adaptive period
    adaptive_length = adaptive_period(close, auto_min, auto_max, auto_avg)

    # Calculate the final adaptive RSI using the adaptive length
    adaptive_rsi[auto_min:] = np.where(
        avg_loss[auto_min:] == 0,
        100,
        100 - (100 / (1 + (avg_gain[auto_min:] / avg_loss[auto_min:]))),
    )

    return (np.array(adaptive_rsi),)


def vhf(
    data: npt.NDArray[np.float64],
    period: cython.int,
) -> npt.NDArray[np.float64]:
    """
    Vertical Horizontal Filter (VHF).

    The VHF is a technical indicator that measures the trend strength of a price series.

    Formula:
    VHF = ABS(pHIGH - pLOW) / SUM(ABS(Pi - Pi-1))

    Parameters:
    - data: A NumPy array of shape (n,) containing price data.
    - period: The period for the VHF calculation.
    - fillna: If True, fill NaN values with 0 (default is False).

    Returns:
    - A NumPy array of shape (n,) containing the VHF values.
    """

    vhf = np.empty_like(data)
    vhf.fill(np.nan)  # Initialize with NaN for non-computable values

    for idx in range(period - 1, len(data)):
        high = np.max(data[idx + 1 - period : idx + 1])
        low = np.min(data[idx + 1 - period : idx + 1])
        price_changes = np.abs(np.diff(data[idx + 1 - period : idx + 1]))

        if np.sum(price_changes) != 0:  # Avoid division by zero
            vhf[idx] = abs(high - low) / np.sum(price_changes)

    # Handle NaN values
    vhf = np.nan_to_num(vhf, nan=0.0)

    return vhf
