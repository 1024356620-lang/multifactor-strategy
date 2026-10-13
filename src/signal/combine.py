 # combine.py
import pandas as pd
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
