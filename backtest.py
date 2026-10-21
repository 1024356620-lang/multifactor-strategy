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
        if not (df['date'] <= rb_date).any():
            continue
            
        # 1. 严格判定入场日：若当天停牌，视为未成交
        entry_row = df[df['date'] == entry_date]
        if entry_row.empty or pd.isna(entry_row['open'].iloc[0]):
            rets[sym] = float('nan')
            continue
            
        open_t1 = entry_row['open'].iloc[0]
        if open_t1 <= 0:
            rets[sym] = float('nan')
            continue
        
        # 2. 获取退出日及之前的子集
        exit_sub = df[(df['date'] >= entry_date) & (df['date'] <= exit_date)]
        if exit_sub.empty:
            rets[sym] = float('nan')
            continue
            
        # 3. 剥离无效的 NaN 收盘价
        valid_closes = exit_sub['close'].dropna()
        valid_closes = valid_closes[valid_closes > 0]
        
        if valid_closes.empty:
            # 买入后到期末连一个有效收盘价都没有（极度异常数据），记 0 或不计入
            rets[sym] = float('nan')
            continue
            
        close_end = valid_closes.iloc[-1]
        rets[sym] = close_end / open_t1 - 1
            
    return rets