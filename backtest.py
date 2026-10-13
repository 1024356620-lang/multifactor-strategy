# backtest.py
import pandas as pd
import numpy as np
from config import FACTOR_WEIGHTS, PERIODS_PER_YEAR

# 从新位置 re-export，保持旧 import 兼容
from src.signal.combine import composite_score
from src.backtest.metrics import performance_summary


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