# backtest.py
import pandas as pd
import numpy as np
from config import FACTOR_WEIGHTS, PERIODS_PER_YEAR


def composite_score(df):
    """按权重合成综合打分（等权加权求和）"""
    score = pd.Series(0.0, index=df.index)
    total_w = 0.0
    for col, w in FACTOR_WEIGHTS.items():
        if col in df.columns:
            score += df[col] * w
            total_w += abs(w)
    return score / total_w if total_w > 0 else score


def calc_period_return(klines, rb_date, next_date):
    """计算每只股票从 rb_date 到 next_date 的收益率"""
    rets = {}
    rb_date = pd.to_datetime(rb_date)
    next_date = pd.to_datetime(next_date)
    for sym, df in klines.items():
        past = df[df['date'] <= rb_date]
        fut = df[(df['date'] > rb_date) & (df['date'] <= next_date)]
        if len(past) == 0 or len(fut) == 0:
            continue
        rets[sym] = fut['close'].iloc[-1] / past['close'].iloc[-1] - 1
    return rets


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

    cum_ret = nav.iloc[-1] - 1
    ann_ret = nav.iloc[-1] ** (periods_per_year / n) - 1

    # 年化波动：单期收益标准差 × sqrt(年化因子)
    period_vol = ls.std(ddof=1)
    ann_vol = period_vol * np.sqrt(periods_per_year)

    # 夏普（无风险利率默认为 0）
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # t 值：检验"平均单期收益是否为 0"
    # t = mean(ls) / (std(ls) / sqrt(n))
    if period_vol > 0:
        t_stat = ls.mean() / (period_vol / np.sqrt(n))
    else:
        t_stat = 0.0

    # 年化 t 值（等价写法，便于和夏普对照）
    # t_ann = sharpe * sqrt(n / periods_per_year)
    t_ann = sharpe * np.sqrt(n / periods_per_year) if ann_vol > 0 else 0.0

    max_dd = (nav / nav.cummax() - 1).min()
    win_rate = (ls > 0).mean()

    return {
        '累计收益': f"{cum_ret:.2%}",
        '年化收益': f"{ann_ret:.2%}",
        '年化波动': f"{ann_vol:.2%}",
        '夏普比率': f"{sharpe:.2f}",
        't值': f"{t_stat:.2f}",
        '样本期数': n,
        '年化因子': periods_per_year,
        '最大回撤': f"{max_dd:.2%}",
        '胜率': f"{win_rate:.1%}",
    }# backtest.py
import pandas as pd
import numpy as np
from config import FACTOR_WEIGHTS, PERIODS_PER_YEAR


def composite_score(df):
    """按权重合成综合打分（等权加权求和）"""
    score = pd.Series(0.0, index=df.index)
    total_w = 0.0
    for col, w in FACTOR_WEIGHTS.items():
        if col in df.columns:
            score += df[col] * w
            total_w += abs(w)
    return score / total_w if total_w > 0 else score


def calc_period_return(klines, rb_date, next_date):
    """计算每只股票从 rb_date 到 next_date 的收益率"""
    rets = {}
    rb_date = pd.to_datetime(rb_date)
    next_date = pd.to_datetime(next_date)
    for sym, df in klines.items():
        past = df[df['date'] <= rb_date]
        fut = df[(df['date'] > rb_date) & (df['date'] <= next_date)]
        if len(past) == 0 or len(fut) == 0:
            continue
        rets[sym] = fut['close'].iloc[-1] / past['close'].iloc[-1] - 1
    return rets


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

    cum_ret = nav.iloc[-1] - 1
    ann_ret = nav.iloc[-1] ** (periods_per_year / n) - 1

    # 年化波动：单期收益标准差 × sqrt(年化因子)
    period_vol = ls.std(ddof=1)
    ann_vol = period_vol * np.sqrt(periods_per_year)

    # 夏普（无风险利率默认为 0）
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # t 值：检验"平均单期收益是否为 0"
    # t = mean(ls) / (std(ls) / sqrt(n))
    if period_vol > 0:
        t_stat = ls.mean() / (period_vol / np.sqrt(n))
    else:
        t_stat = 0.0

    # 年化 t 值（等价写法，便于和夏普对照）
    # t_ann = sharpe * sqrt(n / periods_per_year)
    t_ann = sharpe * np.sqrt(n / periods_per_year) if ann_vol > 0 else 0.0

    max_dd = (nav / nav.cummax() - 1).min()
    win_rate = (ls > 0).mean()

    return {
        '累计收益': f"{cum_ret:.2%}",
        '年化收益': f"{ann_ret:.2%}",
        '年化波动': f"{ann_vol:.2%}",
        '夏普比率': f"{sharpe:.2f}",
        't值': f"{t_stat:.2f}",
        '样本期数': n,
        '年化因子': periods_per_year,
        '最大回撤': f"{max_dd:.2%}",
        '胜率': f"{win_rate:.1%}",
    }