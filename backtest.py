# backtest.py
import pandas as pd
from config import FACTOR_WEIGHTS, PERIODS_PER_YEAR
from src.signal.combine import composite_score
from src.backtest.metrics import performance_summary


def calc_period_return(klines, rb_date, entry_date, exit_date):
    """
    计算持仓期收益：T日收盘信号，T+1日(entry_date)开盘买入，持有至exit_date收盘。
    如果遇到停牌无法在 entry_date 买入，则记为 NaN（由上层剔除或分配权重）。
    """
    rets = {}
    rb_date = pd.to_datetime(rb_date)
    entry_date = pd.to_datetime(entry_date)
    exit_date = pd.to_datetime(exit_date)
    
    for sym, df in klines.items():
        # 确认股票在买入日前有数据
        if not (df['date'] <= rb_date).any():
            continue
            
        # 严格锁定入场日当天的开盘价
        entry_row = df[df['date'] == entry_date]
        if entry_row.empty:
            # 如果入场日停牌，记为 NaN（稍后处理权重）
            rets[sym] = float('nan')
            continue
            
        open_t1 = entry_row['open'].iloc[0]
        
        # 严格锁定退出日当天的收盘价，若退出日停牌，用最近一次有效收盘价
        exit_sub = df[(df['date'] > entry_date) & (df['date'] <= exit_date)]
        if exit_sub.empty:
            rets[sym] = float('nan')
            continue
        close_end = exit_sub['close'].iloc[-1]
        
        if open_t1 > 0:
            rets[sym] = close_end / open_t1 - 1
        else:
            rets[sym] = float('nan')
            
    return rets