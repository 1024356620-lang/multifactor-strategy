# backtest.py
import pandas as pd
import numpy as np
from config import FACTOR_WEIGHTS


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


def performance_summary(nav, ls_series):
    """绩效汇总（季度调仓，年化因子=4）"""
    nav = pd.Series(nav)
    cum_ret = nav.iloc[-1] - 1
    n_periods = len(ls_series)
    ann_ret = nav.iloc[-1] ** (4 / n_periods) - 1 if n_periods > 0 else 0
    ann_vol = ls_series.std() * np.sqrt(4)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (nav / nav.cummax() - 1).min()
    win_rate = (ls_series > 0).mean()
    return {
        '累计收益': f"{cum_ret:.2%}",
        '年化收益': f"{ann_ret:.2%}",
        '年化波动': f"{ann_vol:.2%}",
        '夏普比率': f"{sharpe:.2f}",
        '最大回撤': f"{max_dd:.2%}",
        '月度胜率': f"{win_rate:.1%}",
    }