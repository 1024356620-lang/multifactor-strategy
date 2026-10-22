# anomaly.py
"""异常状态分类与统计。不依赖账本，可单独用。"""
import pandas as pd
from config import (
    STATUS_FILLED, STATUS_UNFILLED_SUSPENDED, STATUS_HELD_SUSPENDED,
    STATUS_DATA_ERROR, STATUS_NO_VALUATION,
)


def classify_symbol(entry_row, exit_closes, ret):
    """
    entry_row: 入场日行（Series 或 None）
    exit_closes: 持仓期有效收盘价 Series
    ret: calc_period_return 返回的收益
    返回状态字符串。
    """
    if entry_row is None or pd.isna(entry_row.get("open", None)):
        return STATUS_UNFILLED_SUSPENDED
    if entry_row["open"] <= 0:
        return STATUS_DATA_ERROR
    if exit_closes is None or len(exit_closes) == 0:
        return STATUS_NO_VALUATION
    if pd.isna(ret):
        return STATUS_HELD_SUSPENDED
    return STATUS_FILLED


def summarize(holdings_csv):
    """输出异常统计表。"""
    df = pd.read_csv(holdings_csv)
    df["abs_w"] = df["target_weight"].abs()
    return df.groupby("status").agg(
        n=("symbol", "count"),
        total_abs_weight=("abs_w", "sum"),
        total_pnl=("pnl_contrib", "sum"),
    ).reset_index()


def data_quality_check(klines, start, end):
    """价格缺失、异常涨跌幅扫描。"""
    issues = []
    for sym, df in klines.items():
        sub = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]
        if sub.empty:
            continue
        miss = sub["close"].isna().sum()
        if miss > 0:
            issues.append({"symbol": sym, "issue": "missing_close", "count": int(miss)})
        if "pctChg" in sub.columns:
            extreme = (sub["pctChg"].abs() > 20).sum()
            if extreme > 0:
                issues.append({"symbol": sym, "issue": "extreme_pctchg", "count": int(extreme)})
    return pd.DataFrame(issues)