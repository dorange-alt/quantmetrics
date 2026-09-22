"""quantmetrics 的 6 个指标实现。"""

import numpy as np

__all__ = [
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "max_drawdown_window",
    "total_return",
]


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


def max_drawdown(prices):
    """最大回撤，返回一个 <= 0 的比例（-0.5 = 从高点跌掉一半）。

    约定：回撤是负数，不做 abs() 转正。理由见 docs/formulas.md。

    思路三步：
      1) peak = 历史高点序列（每个位置记「到目前为止见过的最高价」）
      2) dd   = 价格 / 历史高点 - 1，这个序列恒 <= 0
      3) 取 dd 的最小值，就是跌得最惨的那一次
    """
    p = _as_prices(prices)
    peak = np.maximum.accumulate(p)
    dd = p / peak - 1.0
    return float(dd.min())


def max_drawdown_window(prices):
    """最深回撤的 (峰下标, 谷下标, 深度)。

    第 3 个返回值是负数，与 max_drawdown 的结果一致。

    「索引反推」是这一块唯一的考点：
      先定位谷底 trough = argmin(dd)，
      再 *只在* p[: trough + 1] 这个子区间里找峰。

    为什么不能直接 np.argmax(peak)？因为那是「全序列视角」，
    找的是整个序列最高价第一次出现的位置。若最高价出现在谷底之后
    （跌下去又涨回来、最后创了新高），峰索引就会落到谷底右边，
    逻辑上荒谬。反例：p=[100,120,60,90,130] ——
      错误写法给 (4, 2, -0.5)，正确答案是 (1, 2, -0.5)。
    """
    p = _as_prices(prices)
    peak = np.maximum.accumulate(p)
    dd = p / peak - 1.0

    trough = int(np.argmin(dd))
    peak_idx = int(np.argmax(p[: trough + 1]))
    return (peak_idx, trough, float(dd[trough]))
