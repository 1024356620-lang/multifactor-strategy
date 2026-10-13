 # test_factor.py
import pandas as pd
import numpy as np
from factor_calc import calc_factors_at_date


def test_no_lookahead():
    """
    因子计算只应用 <= end_date 的数据。
    构造一段数据，在 end_date 之后加入极端值，
    因子值不应改变。
    """
    dates = pd.date_range('2020-01-01', periods=60, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'close': np.linspace(10, 20, 60),
        'turn': np.random.rand(60) * 0.05,
    })

    end_date = dates[40]
    f1 = calc_factors_at_date(df, end_date)

    # 在 end_date 之后加极端值
    df2 = df.copy()
    df2.loc[df2['date'] > end_date, 'close'] = 9999
    f2 = calc_factors_at_date(df2, end_date)

    # 因子值应该完全一致
    for key in f1:
        assert abs(f1[key] - f2[key]) < 1e-9, f"{key} 受未来数据影响"


def test_factor_min_history():
    """历史不足时返回 None"""
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'close': np.linspace(10, 20, 10),
        'turn': np.random.rand(10) * 0.05,
    })
    result = calc_factors_at_date(df, dates[-1])
    assert result is None
