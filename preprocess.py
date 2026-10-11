# preprocess.py
import pandas as pd
import numpy as np
from config import FACTOR_COLS, FACTOR_DIRECTION

def _winsorize(series, n_mad=3):
    """
    3倍MAD去极值。
    注意：排名百分位本身对极端值不敏感，但去极值仍可保留，
    避免极端值影响后续可能的其他处理。
    """
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or pd.isna(mad):
        return series
    return series.clip(median - n_mad * mad, median + n_mad * mad)

def _standardize(series):
    """
    改为排名百分位，输出范围 [-0.5, 0.5] 的均匀分布。
    优点：
      1. 彻底消除量纲和厚尾影响，各因子在合成时权重真正可控
      2. 对极端值天然免疫，无需担心标准化后仍有超大值
      3. 业界多因子合成的标准做法
    """
    return series.rank(pct=True) - 0.5

def preprocess_factors(df):
    """
    去极值 + 排名百分位 + 方向统一。
    返回处理后的 DataFrame，或 None（数据不足）
    """
    df = df.copy()
    
    # 检查列是否存在
    for col in FACTOR_COLS:
        if col not in df.columns:
            print(f"  缺少因子列: {col}")
            return None
        if df[col].isna().all():
            print(f"  因子 {col} 全为 NaN")
            return None
    
    # 用中位数填充少量 NaN
    for col in FACTOR_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # 去极值 + 排名百分位
    for col in FACTOR_COLS:
        df[col] = _winsorize(df[col])
        df[col] = _standardize(df[col])
    
        # 方向统一
    for col, direction in FACTOR_DIRECTION.items():
        if direction == -1:
            df[col] = -df[col]
    
    return df