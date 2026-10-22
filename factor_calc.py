# factor_calc.py
import pandas as pd
import numpy as np
from config import FACTOR_MIN_HISTORY

def calc_factors_at_date(df, end_date):
    """
    在指定日期计算单只股票的因子值。
    严格只用 <= end_date 的数据，避免未来函数。
    返回 dict 或 None（历史不足）
    """
    sub = df[df['date'] <= pd.to_datetime(end_date)].reset_index(drop=True)
    if len(sub) < FACTOR_MIN_HISTORY:
        return None
    
    sub = sub.copy()
    sub['ret'] = sub['close'].pct_change()
    
    reversal = -(sub['close'].iloc[-1] / sub['close'].iloc[-21] - 1)
    volatility = sub['ret'].iloc[-20:].std() * np.sqrt(252)
    avg_turn = sub['turn'].iloc[-20:].mean()
    
    ma5 = sub['close'].rolling(5).mean().iloc[-1]
    ma20 = sub['close'].rolling(20).mean().iloc[-1]
    ma_ratio = ma5 / ma20 - 1 if ma20 > 0 else 0
    
    
    return {
        'reversal': reversal,
        'volatility': volatility,
        'avg_turn': avg_turn,
        'ma_ratio': ma_ratio,
    }

def build_cross_section(klines, end_date):
    """
    构建某个调仓日的因子截面。
    返回 DataFrame 或 None
    """
    rows = []
    for sym, df in klines.items():
        f = calc_factors_at_date(df, end_date)
        if f:
            f['symbol'] = sym
            rows.append(f)
    if not rows:
        return None
    return pd.DataFrame(rows)