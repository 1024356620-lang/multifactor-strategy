# test_metrics.py
import numpy as np
from src.backtest.metrics import performance_summary


def test_sharpe_known_value():
    """
    手工构造一个已知夏普的序列。
    nav 按季度复利增长，ls 每期固定为 0.05。
    波动为 0 时夏普应为 0（代码里 ann_vol > 0 才计算）。
    """
    ls = [0.05] * 4
    nav = [1.0]
    for r in ls:
        nav.append(nav[-1] * (1 + r))

    result = performance_summary(nav, ls, periods_per_year=4)
    # 波动为 0，夏普应为 0
    assert result['夏普比率'] == '0.00'
    assert result['样本期数'] == 4


def test_sharpe_normal_case():
    """有波动的正常情况，t 值应可计算"""
    ls = [0.05, -0.02, 0.03, 0.01]
    nav = [1.0]
    for r in ls:
        nav.append(nav[-1] * (1 + r))

    result = performance_summary(nav, ls, periods_per_year=4)
    assert result['样本期数'] == 4
    assert 't值' in result
    # t 值应该是个有限数
    assert float(result['t值']) != 0