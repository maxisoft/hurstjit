#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Hurst exponent and RS-analysis
https://en.wikipedia.org/wiki/Hurst_exponent
https://en.wikipedia.org/wiki/Rescaled_range
"""

name = "hurstjit"
__version__ = '0.0.6'
import numpy as np
import numba


@numba.njit(fastmath=True, inline='always')
def _std_ddof(a, ddof=0):
    """Workaround for numba std(ddof=?) missing parameter"""
    res = np.std(a)
    if ddof != 0:
        if ddof == len(a):
            res *= 0
        else:
            res *= len(a) / (len(a) - ddof)
    return res


@numba.njit(fastmath=True, inline='always')
def __to_inc(x):
    incs = x[1:] - x[:-1]
    return incs


@numba.njit(fastmath=True, inline='always')
def __to_pct(x):
    pcts = x[1:] / x[:-1] - 1.
    return pcts


@numba.njit(fastmath=True, inline='always')
def __get_simplified_RS(series, kind):
    """
    Simplified version of rescaled range
    Parameters
    ----------
    series : array-like
        (Time-)series
    kind : str
        The kind of series (refer to compute_Hc docstring)
    """

    if kind == 'random_walk':
        incs = __to_inc(series)
        R = np.amax(series) - np.amin(series)  # range in absolute values
        S = _std_ddof(incs, ddof=1)
    elif kind == 'price':
        pcts = __to_pct(series)
        R = np.amax(series) / np.amin(series) - 1.  # range in percent
        S = _std_ddof(pcts, ddof=1)
    elif kind == 'change':
        incs = series
        _series = np.zeros(len(incs) + 1, dtype=series.dtype)
        _series[1:] = np.cumsum(incs)
        R = np.amax(_series) - np.amin(_series)  # range in absolute values
        S = _std_ddof(incs, ddof=1)

    if S == 0:
        return 0  # return 0 to skip this interval due the undefined R/S ratio

    return R / S


@numba.njit(fastmath=True)
def __get_RS(series, kind):
    """
    Get rescaled range (using the range of cumulative sum
    of deviations instead of the range of a series as in the simplified version
    of R/S) from a time-series of values.
    Parameters
    ----------
    series : array-like
        (Time-)series
    kind : str
        The kind of series (refer to compute_Hc docstring)
    """

    if kind == 'random_walk':
        incs = __to_inc(series)
        mean_inc = (series[-1] - series[0]) / len(incs)
        deviations = incs - mean_inc
        Z = np.cumsum(deviations)
        R = np.amax(Z) - np.amin(Z)
        S = _std_ddof(incs, ddof=1)

    elif kind == 'price':
        incs = __to_pct(series)
        mean_inc = np.sum(incs) / len(incs)
        deviations = incs - mean_inc
        Z = np.cumsum(deviations)
        R = np.amax(Z) - np.amin(Z)
        S = _std_ddof(incs, ddof=1)

    elif kind == 'change':
        incs = series
        mean_inc = np.sum(incs) / len(incs)
        deviations = incs - mean_inc
        Z = np.cumsum(deviations)
        R = np.amax(Z) - np.amin(Z)
        S = _std_ddof(incs, ddof=1)

    if S == 0:
        return 0  # return 0 to skip this interval due undefined R/S

    return R / S


@numba.njit(fastmath=True, inline='always')
def _log(a, base):
    if base == 10:
        return np.log10(a)
    elif base == 2:
        return np.log2(a)
    elif base is None:
        return np.log(a)
    return np.log(a) / np.log(base)


@numba.njit(fastmath=True, inline='always')
def _exp(a, base):
    if base is None:
        return np.exp(a)
    return base ** a


@numba.njit(fastmath=True)
def compute_Hc(series, kind='random_walk', min_window=10, max_window=-1, simplified=True, base=10, increment=0.25):
    """
    Compute H (Hurst exponent) and C according to Hurst equation:
    E(R/S) = c * T^H
    Refer to:
    https://en.wikipedia.org/wiki/Hurst_exponent
    https://en.wikipedia.org/wiki/Rescaled_range
    https://en.wikipedia.org/wiki/Random_walk
    Parameters
    ----------
    series : array-like
        (Time-)series
    kind : str
        Kind of series
        possible values are 'random_walk', 'change' and 'price':
        - 'random_walk' means that a series is a random walk with random increments;
        - 'price' means that a series is a random walk with random multipliers;
        - 'change' means that a series consists of random increments
            (thus produced random walk is a cumulative sum of increments);
    min_window : int, default 10
        the minimal window size for R/S calculation
    max_window : int, default is the length of series minus 1
        the maximal window size for R/S calculation
    simplified : bool, default True
        whether to use the simplified or the original version of R/S calculation
    base : float, default 10
        the base to compute logs
        use None to use natural log
    increment: float, default 0.25
        parameter to control precision / computation time
        a lower value mean better precision but longer computation time
    Returns tuple of
        H, c and data
        where H and c — parameters or Hurst equation
        and data is a tuple of 2 arrays: time intervals and R/S-values for correspoding time interval
        for further plotting log(data[0]) on X and log(data[1]) on Y
    """
    if min_window is None:
        min_window = base if base is not None else 0
        min_window = max(min_window, 8)

    if min_window < 2:
        raise ValueError("min_window need to be greater than 2 in order to compute revelant variances")

    if len(series) < _exp(2, base):
        raise ValueError("Series length must be greater or equal to base ** 2")

    elif np.isnan(np.min(series)):
        raise ValueError("Series contains NaNs")

    if max_window < 0:
        max_window = len(series) + max_window

    if max_window <= min_window:
        raise ValueError("max_window incompatible with min_window")

    window_sizes = (_exp(np.arange(_log(min_window, base), _log(max_window, base), increment), base)).astype(np.int64)
    if window_sizes[-1] < len(series):
        window_sizes = np.append(window_sizes, (len(series),))
    window_sizes = np.unique(window_sizes)
    dtype = np.float64  # TODO find a way to use float32 too
    RS = np.zeros(len(window_sizes), dtype=dtype)
    for i in numba.prange(len(window_sizes)):
        w = window_sizes[i]
        if w <= 0:
            continue
        acc = dtype(0)
        n = np.int64(0)
        for start in range(0, len(series), w):
            if start + w > len(series):
                break
            if simplified:
                tmp = __get_simplified_RS(series[start:start + w], kind)
            else:
                tmp = __get_RS(series[start:start + w], kind)
            if tmp != 0:
                acc += tmp
                n += 1
        if n > 0:
            RS[i] = acc / n

    A = np.ones((len(window_sizes), 2), dtype=dtype)
    A[:, 0] = _log(window_sizes, base)
    H, c = np.linalg.lstsq(A, _log(RS, base), rcond=-1)[0]

    c = _exp(c, base)
    return H, c, (window_sizes, RS)


@numba.njit(fastmath=True)
def random_walk(length, proba=0.5, min_lookback=1, max_lookback=100, cumprod=False):
    """
    Generates a random walk series

    Parameters
    ----------

    proba : float, default 0.5
        the probability that the next increment will follow the trend.
        Set proba > 0.5 for the persistent random walk,
        set proba < 0.5 for the antipersistent one

    min_lookback: int, default 1
    max_lookback: int, default 100
        minimum and maximum window sizes to calculate trend direction
    cumprod : bool, default False
        generate a random walk as a cumulative product instead of cumulative sum
    """

    assert (min_lookback >= 1)
    assert (max_lookback >= min_lookback)

    if max_lookback > length:
        max_lookback = length

    if not cumprod:  # ordinary increments
        series = np.zeros(length, dtype=np.float32)  # array of prices

        for i in range(1, length):
            if i < min_lookback + 1:
                direction = np.sign(np.random.randn())
            else:
                lookback = np.random.randint(min_lookback, min(i - 1, max_lookback) + 1)
                direction = np.sign(series[i - 1] - series[i - 1 - lookback]) * np.sign(
                    proba - np.random.uniform(0.0, 1.0))
            series[i] = series[i - 1] + np.fabs(np.random.randn()) * direction
    else:  # percent changes
        series = np.ones(length, dtype=np.float32)  # array of prices
        for i in range(1, length):
            if i < min_lookback + 1:
                direction = np.sign(np.random.randn())
            else:
                lookback = np.random.randint(min_lookback, min(i - 1, max_lookback) + 1)
                direction = np.sign(series[i - 1] / series[i - 1 - lookback] - 1.) * np.sign(
                    proba - np.random.uniform(0.0, 1.0))
            series[i] = series[i - 1] * np.fabs(1 + np.random.randn() / 1000. * direction)

    return series


if __name__ == '__main__':

    # Use random_walk() function or generate a random walk series manually:
    # series = random_walk(99999, cumprod=True)
    np.random.seed(42)
    random_changes = 1. + np.random.randn(99999) / 1000.
    series = np.cumprod(random_changes)  # create a random walk from random changes

    # Evaluate Hurst equation
    H, c, data = compute_Hc(series, kind='price', simplified=True)

    # Plot
    # uncomment the following to make a plot using Matplotlib:
    """
    import matplotlib.pyplot as plt

    f, ax = plt.subplots()
    ax.plot(data[0], c*data[0]**H, color="deepskyblue")
    ax.scatter(data[0], data[1], color="purple")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time interval')
    ax.set_ylabel('R/S ratio')
    ax.grid(True)
    plt.show()
    """

    print("H={:.4f}, c={:.4f}".format(H,c))
    assert H<0.6 and H>0.4

