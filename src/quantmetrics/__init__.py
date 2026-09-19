"""quantmetrics —— 纯 numpy 实现的量化风险指标库。

对外接口在各指标写完后从这里导出。当前顺序：
  第 1 步：在 metrics.py 里实现 total_return / annualized_return / annualized_volatility
  第 2 步：补 max_drawdown / max_drawdown_window / sharpe_ratio
  第 3 步：把 6 个函数在这里 import 出来并写进 __all__

导出后长这样（现在还不要写，等函数真的存在了再加，否则 import 会报错）：

    from .metrics import (
        annualized_return,
        annualized_volatility,
        max_drawdown,
        max_drawdown_window,
        sharpe_ratio,
        total_return,
    )

    __all__ = [
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "max_drawdown_window",
        "sharpe_ratio",
        "total_return",
    ]
"""
