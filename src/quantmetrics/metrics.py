"""quantmetrics 的 6 个指标实现。"""

import numpy as np

__all__ = ["annualized_return", "annualized_volatility", "total_return"]


def _as_prices(prices):
    """把输入统一成 float 一维数组，并做长度校验。"""
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1:
        raise ValueError("prices must be one-dimensional")
    if p.size < 2:
        raise ValueError("prices must contain at least two observations")
    return p


def _returns(prices):
    """由价格序列算出每期简单收益率，长度比价格少 1。"""
    p = _as_prices(prices)
    r = p[1:] / p[:-1] - 1.0
    if r.size < 2:
        raise ValueError("need at least two returns to compute std with ddof=1")
    return r


def total_return(prices):
    p = _as_prices(prices)
    return float(p[-1] / p[0] - 1.0)


def annualized_return(prices, ppy=252):
    p = _as_prices(prices)
    n = p.size - 1
    return float((1.0 + total_return(p)) ** (ppy / n) - 1.0)


def annualized_volatility(prices, ppy=252):
    r = _returns(prices)
    return float(np.std(r, ddof=1) * np.sqrt(ppy))
