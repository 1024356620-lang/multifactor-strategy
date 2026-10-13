# test_preprocess.py
import pandas as pd
import numpy as np
from preprocess import _standardize, _winsorize


def test_standardize_range():
    """排名百分位输出应在 [-0.5, 0.5]"""
    s = pd.Series(np.random.randn(100))
    out = _standardize(s)
    assert out.min() >= -0.5
    assert out.max() <= 0.5


def test_winsorize_clips_outlier():
    """3倍MAD应把极端值拉回"""
    s = pd.Series([1, 2, 3, 4, 5, 1000])
    out = _winsorize(s)
    assert out.max() < 1000