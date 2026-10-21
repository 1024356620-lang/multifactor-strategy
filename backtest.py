# backtest.py
import pandas as pd
from config import FACTOR_WEIGHTS, PERIODS_PER_YEAR

# Re-export 保持向下兼容，避免其他地方旧引用报错
from src.signal.combine import composite_score
from src.backtest.metrics import performance_summary


def calc_period_return(klines, rb_date, next_date):
    """
    计算持仓期收益：T日收盘产生信号，T+1日开盘买入，持有至下期调仓日收盘。
    对停牌或缺失数据的股票进行跳过处理，由上层计算等权平均时忽略。
    """
    rets = {}
    rb_date = pd.to_datetime(rb_date)
    next_date = pd.to_datetime(next_date)
    
    for sym, df in klines.items():
        past = df[df['date'] <= rb_date]
        if past.empty:
            continue
            
        # 获取调仓日之后（即 T+1 日及以后）的行情
        fut = df[(df['date'] > rb_date) & (df['date'] <= next_date)]
        if fut.empty:
            continue
            
        open_t1 = fut['open'].iloc[0]
        close_end = fut['close'].iloc[-1]
        
        if open_t1 > 0:
            rets[sym] = close_end / open_t1 - 1
            
    return rets