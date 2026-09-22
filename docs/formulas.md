# quantmetrics 指标公式与接口约定

## 接口签名

```python
def total_return(prices: np.ndarray) -> float
def annualized_return(prices: np.ndarray, ppy: int = 252) -> float
def annualized_volatility(prices: np.ndarray, ppy: int = 252) -> float
def sharpe_ratio(prices: np.ndarray, rf: float = 0.0, ppy: int = 252) -> float
def max_drawdown(prices: np.ndarray) -> float
def max_drawdown_window(prices: np.ndarray) -> tuple[int, int, float]
```

## 全局设定

- `prices` 是**价格序列**，不是收益率序列；函数内部自己算差分。
- `ppy` = periods per year，每年周期数。日频 252 / 周频 52 / 月频 12。
- `rf` 是**年化**无风险利率，用前必须先除以 `ppy` 换算到每期。

## 六个量的三问（定义 / 单位 / 边界）

### total_return

- **定义**：末价 ÷ 首价 − 1，即 `R = P_N / P_0 − 1`
- **单位**：无量纲比例（`0.21` 表示 +21%）
- **边界**：长度 < 2 抛 `ValueError`

### annualized_return

- **定义**：把总收益率按每期复利外推到一年：`(1 + R) ** (ppy / n) − 1`，其中 `n = len(prices) − 1`
- **单位**：年化比例
- **边界**：同 `total_return`。注意 `ppy` 必须与序列的真实频率匹配 —— 4 期数据按 252 外推会得到 +1083% 这种夸张值，那不是 bug，是公式在数据太短时的必然表现

### annualized_volatility

- **定义**：每期收益率的标准差 × `√ppy`，用 `ddof=1`（样本标准差）
- **单位**：年化比例
- **边界**：需要至少 2 个**收益率**（即至少 3 个价格点）才有样本标准差；收益率完全恒定时标准差为 0，返回 `0.0`

### max_drawdown

- **定义**：峰到谷的最大跌幅。先造历史高点序列 `peak = np.maximum.accumulate(prices)`（每个位置记「到目前为止见过的最高价」），再算回撤序列 `dd = prices / peak − 1`，取 `dd.min()`
- **单位**：无量纲比例，**恒 ≤ 0**（`-0.5` 表示从高点跌掉一半）
- **约定（重要）**：**不做 `abs()` 转正**。回撤为负是这个库的明确约定
- **边界**：长度 < 2 抛 `ValueError`（**方案 A**：与收益类函数保持一致，调用方只需记住一条规则）

### max_drawdown_window

- **定义**：最深回撤的 `(峰下标, 谷下标, 深度)` 三元组；第 3 个值与同数据下 `max_drawdown` 的结果相等
- **单位**：前两个是**索引位置**（整数下标，从 0 开始），第三个是比例（≤ 0）
- **关键实现细节**：不能直接对整条 `peak` 取 `np.argmax` —— 那是全序列视角，找的是整个序列最高价第一次出现的位置，若最高价出现在谷底**之后**（跌下去又涨回来创新高），峰索引会落到谷底右边。**必须先定位谷底 `t = argmin(dd)`，再只在 `p[:t+1]` 子区间里找峰。** 反例：`[100, 120, 60, 90, 130]` → 错误写法给 `(4, 2, -0.5)`，正确答案 `(1, 2, -0.5)`
- **边界**：长度 < 2 抛 `ValueError`；单调上涨时回撤为 0，返回 `(0, 0, 0.0)`

### sharpe_ratio

- **定义**：`(mean(r) − rf / ppy) / std(r, ddof=1) × √ppy`，其中 `r` 是每期简单收益率序列
- **单位**：无量纲（风险调整后收益，越大越好）
- **边界**：
  - 长度不足抛 `ValueError`（走 `_returns` 的校验）
  - **收益率完全恒定（`std = 0`）时抛 `ValueError`** —— 这是本库明确的设计取舍：**fail fast**。调用方应该立刻知道「这批数据算不出夏普」，而不是拿到一个 `nan` 把它静默带到下游，最后在不相干的地方爆出难以定位的错误
- **关键实现细节**：`rf` 是**年化**无风险利率，必须先除以 `ppy` 换算到每期。若写成 `mean(r) − rf`，等于每一期都减掉一整个年化利率（日频数据每期收益只有 `0.0x%` 量级），结果会离谱到看不出错
