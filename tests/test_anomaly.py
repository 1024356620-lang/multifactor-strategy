# tests/test_anomaly.py
import pandas as pd
import numpy as np
from anomaly import classify_symbol
from config import (
    STATUS_FILLED, STATUS_UNFILLED_SUSPENDED, STATUS_HELD_SUSPENDED,
    STATUS_DATA_ERROR, STATUS_NO_VALUATION,
)


# 1. 未成交（停牌无行） 
def test_unfilled_suspended_no_row():
    """entry_row 为 None（T+1 停牌，数据里没有该行）→ unfilled_suspended"""
    assert classify_symbol(None, pd.Series([10.0]), 0.1) == STATUS_UNFILLED_SUSPENDED


# 2. 未成交（open 为 NaN） 
def test_unfilled_nan_open():
    """entry_row 存在但 open 为 NaN（停牌但有行）→ unfilled_suspended"""
    row = pd.Series({"open": np.nan})
    assert classify_symbol(row, pd.Series([10.0]), 0.1) == STATUS_UNFILLED_SUSPENDED


# 3. 数据错误（open <= 0）
def test_data_error_open_zero():
    """open 为 0 → data_error"""
    row = pd.Series({"open": 0.0})
    assert classify_symbol(row, pd.Series([10.0]), 0.1) == STATUS_DATA_ERROR


def test_data_error_open_negative():
    """open 为负 → data_error"""
    row = pd.Series({"open": -1.0})
    assert classify_symbol(row, pd.Series([10.0]), 0.1) == STATUS_DATA_ERROR


# 4. 无估值（买入后无有效收盘价）
def test_no_valuation_empty_closes():
    """exit_closes 为空 → no_valuation"""
    row = pd.Series({"open": 10.0})
    assert classify_symbol(row, pd.Series([], dtype=float), np.nan) == STATUS_NO_VALUATION


def test_no_valuation_none_closes():
    """exit_closes 为 None → no_valuation"""
    row = pd.Series({"open": 10.0})
    assert classify_symbol(row, None, np.nan) == STATUS_NO_VALUATION


# 5. 持仓停牌（有收盘价但收益 NaN） 
def test_held_suspended():
    """entry 成交、exit_closes 非空，但 ret 为 NaN → held_suspended"""
    row = pd.Series({"open": 10.0})
    assert classify_symbol(row, pd.Series([10.0]), np.nan) == STATUS_HELD_SUSPENDED


# 6. 正常成交
def test_filled():
    """entry 成交、exit_closes 非空、ret 非 NaN → filled"""
    row = pd.Series({"open": 10.0})
    assert classify_symbol(row, pd.Series([11.0]), 0.1) == STATUS_FILLED


# 7. 边界：entry_row 为 None 但 exit_closes 也空 
def test_unfilled_takes_priority_over_no_valuation():
    """未成交优先于无估值：entry 停牌时，即使 exit 也空，仍判 unfilled"""
    assert classify_symbol(None, pd.Series([], dtype=float), np.nan) == STATUS_UNFILLED_SUSPENDED