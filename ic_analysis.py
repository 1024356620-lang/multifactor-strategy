# ic_analysis.py
import pandas as pd
import numpy as np
from config import FACTOR_COLS

def calc_ic(factor_df, future_ret_col='future_ret'):
    """
    计算单期各因子的 RankIC。
    返回 dict {factor: ic}
    """
    result = {}
    for col in FACTOR_COLS:
        if col not in factor_df.columns:
            continue
        sub = factor_df[[col, future_ret_col]].dropna()
        if len(sub) < 10:
            result[col] = np.nan
            continue
        result[col] = sub[col].corr(sub[future_ret_col], method='spearman')
    return result

def summarize_ic(ic_records):
    """
    ic_records: list of dict，每期一条，含 date + 各因子 IC
    返回汇总 DataFrame
    """
    df = pd.DataFrame(ic_records)
    if df.empty:
        return pd.DataFrame()
    
    rows = []
    for col in FACTOR_COLS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        rows.append({
            'factor': col,
            'IC均值': s.mean(),
            'IC_IR': s.mean() / s.std() if s.std() > 0 else np.nan,
            '胜率': (s > 0).mean(),
            '样本数': len(s),
        })
    return pd.DataFrame(rows)