 # layered.py
import pandas as pd
from config import FACTOR_COLS
from src.signal.combine import composite_score


def layered_returns(factor_df, future_rets, n_groups=5):
    """按综合打分分层，返回每层平均收益"""
    df = factor_df.copy()
    df['score'] = composite_score(df)
    df['future_ret'] = df['symbol'].map(future_rets)
    df = df.dropna(subset=['future_ret']).sort_values('score', ascending=False)
    if len(df) < n_groups * 5:
        return None
    df['group'] = pd.qcut(
        df['score'].rank(method='first'),
        n_groups,
        labels=[f'G{i+1}' for i in range(n_groups)]
    )
    return df.groupby('group')['future_ret'].mean()


def single_factor_layered(factor_df, future_rets, n_groups=5):
    """
    对每个因子单独分层，看单调性。
    返回 DataFrame：index=因子名，columns=G1..G5
    """
    results = {}
    df = factor_df.copy()
    df['future_ret'] = df['symbol'].map(future_rets)
    df = df.dropna(subset=['future_ret'])

    for col in FACTOR_COLS:
        if col not in df.columns:
            continue
        sub = df[[col, 'future_ret']].dropna().copy()
        if len(sub) < n_groups * 5:
            continue
        sub['group'] = pd.qcut(
            sub[col].rank(method='first'), n_groups,
            labels=[f'G{i+1}' for i in range(n_groups)]
        )
        results[col] = sub.groupby('group')['future_ret'].mean()
    return pd.DataFrame(results).T


def check_monotonic(series):
    """判断分层收益是否单调，返回 '递增' / '递减' / '非单调'"""
    vals = series.dropna().values
    if len(vals) < 2:
        return '数据不足'
    if all(vals[i] <= vals[i+1] for i in range(len(vals)-1)):
        return '递增'
    if all(vals[i] >= vals[i+1] for i in range(len(vals)-1)):
        return '递减'
    return '非单调'
