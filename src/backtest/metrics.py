# src/backtest/metrics.py
import pandas as pd
import numpy as np
from config import PERIODS_PER_YEAR


def performance_summary(nav, ls_series, periods_per_year=PERIODS_PER_YEAR):
    """
    绩效汇总。
    参数：
        nav: 净值序列（长度 = 期数 + 1）
        ls_series: 每期多空收益序列
        periods_per_year: 年化因子（季度=4，月度=12，日频=252）
    返回：
        dict，含夏普、t值、样本期数、年化因子
    """
    nav = pd.Series(nav).reset_index(drop=True)
    ls = pd.Series(ls_series).dropna().reset_index(drop=True)
    n = len(ls)

    if n == 0:
        return {'错误': '无有效样本'}

    # 1. 净值相关指标
    cum_ret = nav.iloc[-1] - 1
    ann_ret = nav.iloc[-1] ** (periods_per_year / n) - 1 # 几何年化收益 (CAGR)
    max_dd = (nav / nav.cummax() - 1).min()

    # 2. 收益与波动相关指标（算术平均）
    mean_ret = ls.mean()
    period_vol = ls.std(ddof=1)
    ann_vol = period_vol * np.sqrt(periods_per_year)

    # 3. 严格口径的夏普比率 (算术均值 / 标准差 * sqrt(N))
    # 此处假设无风险利率为 0
    if period_vol > 0:
        sharpe = (mean_ret / period_vol) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    # 4. t 统计量
    if period_vol > 0:
        t_stat = mean_ret / (period_vol / np.sqrt(n))
    else:
        t_stat = 0.0

    win_rate = (ls > 0).mean()

    return {
        '累计收益': f"{cum_ret:.2%}",
        '年化收益': f"{ann_ret:.2%}",
        '年化波动': f"{ann_vol:.2%}",
        '夏普比率': f"{sharpe:.2f}",  # 标准算术夏普
        't值': f"{t_stat:.2f}",
        '样本期数': n,
        '年化因子': periods_per_year,
        '最大回撤': f"{max_dd:.2%}",
        '胜率': f"{win_rate:.1%}",
    }