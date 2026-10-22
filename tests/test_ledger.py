# tests/test_ledger.py
import pandas as pd
import numpy as np
import pytest
from ledger import Ledger
from config import (
    STATUS_FILLED, STATUS_UNFILLED_SUSPENDED, STATUS_NO_VALUATION,
)


def test_reconcile_matches():
    lg = Ledger()
    lg.record_period("2025-09-30", "2025-10-01", "2025-12-31",
                     ["000001"], ["000002"], {"000001": 0.5, "000002": -0.5},
                     {"000001": 10.0, "000002": 20.0},
                     {"000001": 11.0, "000002": 18.0},
                     {"000001": 0.10, "000002": -0.10},
                     {"000001": "filled", "000002": "filled"})
    rec = lg.reconcile()
    gross = rec["gross_pnl"].iloc[0]
    assert abs(gross - (0.5 * 0.10 + (-0.5) * (-0.10))) < 1e-9
    assert abs(rec["net_pnl"].iloc[0] - (gross - rec["total_cost"].iloc[0])) < 1e-9


def test_no_valuation_raises_error():
    """测试：一旦出现缺估值，必须明确报错并停止生成绩效指标，严禁静默算作零收益。"""
    lg = Ledger()
    lg.record_period("2025-09-30", "2025-10-01", "2025-12-31",
                     ["000003"], [], {"000003": 0.5},
                     {"000003": 10.0}, {"000003": float("nan")},
                     {"000003": float("nan")},
                     {"000003": "no_valuation"})
    
    with pytest.raises(ValueError, match="致命错误：发现 1 条缺估值持仓"):
        lg.reconcile()


def test_unfilled_not_counted():
    lg = Ledger()
    lg.record_period("2025-07-01", "2025-07-02", "2025-09-30",
                     [], ["000001"], {"000001": -0.5},
                     {"000001": np.nan}, {"000001": np.nan},
                     {"000001": np.nan},
                     {"000001": "unfilled_suspended"})
    rec = lg.reconcile()
    assert rec["n_unfilled"].iloc[0] == 1
    assert rec["gross_pnl"].iloc[0] == 0.0


def test_one_open_one_close_per_position():
    lg = Ledger()
    lg.record_period("2025-07-01", "2025-07-02", "2025-09-30",
                     ["000001"], ["000002"],
                     {"000001": 0.5, "000002": -0.5},
                     {"000001": 10.0, "000002": 20.0},
                     {"000001": 11.0, "000002": 18.0},
                     {"000001": 0.1, "000002": -0.1},
                     {"000001": "filled", "000002": "filled"})
    trades = pd.DataFrame(lg.trades)
    assert len(trades) == 4
    for sym in ["000001", "000002"]:
        t = trades[trades["symbol"] == sym]
        assert len(t[t["trade_type"] == "open"]) == 1
        assert len(t[t["trade_type"] == "close"]) == 1


def test_open_close_dates_and_prices_match_holding():
    lg = Ledger()
    lg.record_period("2025-07-01", "2025-07-02", "2025-09-30",
                     ["000001"], [], {"000001": 0.5},
                     {"000001": 10.0}, {"000001": 11.0},
                     {"000001": 0.1}, {"000001": "filled"})
    trades = pd.DataFrame(lg.trades)
    o = trades[trades["trade_type"] == "open"].iloc[0]
    c = trades[trades["trade_type"] == "close"].iloc[0]
    assert str(o["entry_date"])[:10] == "2025-07-02"
    assert abs(o["price"] - 10.0) < 1e-9
    assert str(c["entry_date"])[:10] == "2025-09-30"
    assert abs(c["price"] - 11.0) < 1e-9


def test_close_amount_scales_with_return():
    lg = Ledger()
    lg.record_period("2025-07-01", "2025-07-02", "2025-09-30",
                     ["000001"], [], {"000001": 0.5},
                     {"000001": 10.0}, {"000001": 15.0},
                     {"000001": 0.5}, {"000001": "filled"})
    trades = pd.DataFrame(lg.trades)
    o = trades[trades["trade_type"] == "open"].iloc[0]
    c = trades[trades["trade_type"] == "close"].iloc[0]
    assert abs(o["amount"] - 0.5) < 1e-9
    assert abs(c["amount"] - 0.5 * 1.5) < 1e-9


def test_open_direction_long_buy_short_sell():
    lg = Ledger()
    lg.record_period("2025-07-01", "2025-07-02", "2025-09-30",
                     ["000001"], ["000002"],
                     {"000001": 0.5, "000002": -0.5},
                     {"000001": 10.0, "000002": 20.0},
                     {"000001": 11.0, "000002": 18.0},
                     {"000001": 0.1, "000002": -0.1},
                     {"000001": "filled", "000002": "filled"})
    trades = pd.DataFrame(lg.trades)
    o1 = trades[(trades["symbol"] == "000001") & (trades["trade_type"] == "open")].iloc[0]
    o2 = trades[(trades["symbol"] == "000002") & (trades["trade_type"] == "open")].iloc[0]
    assert o1["direction"] == "buy"
    assert o2["direction"] == "sell"


def test_close_direction_long_sell_short_buy_to_cover():
    lg = Ledger()
    lg.record_period("2025-07-01", "2025-07-02", "2025-09-30",
                     ["000001"], ["000002"],
                     {"000001": 0.5, "000002": -0.5},
                     {"000001": 10.0, "000002": 20.0},
                     {"000001": 11.0, "000002": 18.0},
                     {"000001": 0.1, "000002": -0.1},
                     {"000001": "filled", "000002": "filled"})
    trades = pd.DataFrame(lg.trades)
    c1 = trades[(trades["symbol"] == "000001") & (trades["trade_type"] == "close")].iloc[0]
    c2 = trades[(trades["symbol"] == "000002") & (trades["trade_type"] == "close")].iloc[0]
    assert c1["direction"] == "sell"
    assert c2["direction"] == "buy_to_cover"