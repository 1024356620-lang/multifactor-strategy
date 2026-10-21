# backtest.py
import pandas as pd
from config import FACTOR_WEIGHTS, PERIODS_PER_YEAR
from src.signal.combine import composite_score
from src.backtest.metrics import performance_summary


def calc_period_return(klines, rb_date, entry_date, exit_date):
    """
    计算持仓期收益。
    返回 dict: {symbol: return}
    - 如果 entry_date 停牌买不到，返回 NaN（资金留作现金，不承担风险）
    - 如果买入后停牌，则取 exit_date 及之前的最后一次有效收盘价进行盯市估值
    """
    rets = {}
    rb_date = pd.to_datetime(rb_date)
    entry_date = pd.to_datetime(entry_date)
    exit_date = pd.to_datetime(exit_date)
    
    for sym, df in klines.items():
        # 确保股票在产生信号前有历史数据
        if not (df['date'] <= rb_date).any():
            continue
            
        # 严格判定入场日：若当天停牌，视为未成交
        entry_row = df[df['date'] == entry_date]
        if entry_row.empty:
            rets[sym] = float('nan')
            continue
            
        open_t1 = entry_row['open'].iloc[0]
        
        # 寻找退出日及之前最后一个有效收盘价
        exit_sub = df[(df['date'] >= entry_date) & (df['date'] <= exit_date)]
        if exit_sub.empty:
            rets[sym] = float('nan')
            continue
            
        close_end = exit_sub['close'].iloc[-1]
        
        if open_t1 > 0:
            rets[sym] = close_end / open_t1 - 1
        else:
            rets[sym] = float('nan')
            
    return rets