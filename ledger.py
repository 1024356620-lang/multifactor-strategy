# ledger.py
"""
交易账本：每条持仓生成一对开平仓交易。

交易模型：每季全部平仓重建。
- 每条持仓：entry_date 建仓，exit_date 平仓。
- 多头：buy → sell；空头：sell → buy_to_cover。
- 开仓金额 = abs(target_weight)，平仓金额 = abs(target_weight) × (1 + ret)。
  平仓金额按"名义金额随价格变化"计算，更接近真实成交额。
- 费用：开仓 + 平仓各计一次，按金额 × COMMISSION_RATE。
"""
import os
import pandas as pd
import numpy as np
from config import (
    LEDGER_DIR, COMMISSION_RATE, STAMP_TAX_RATE, SLIPPAGE_BPS,
    STATUS_FILLED, STATUS_UNFILLED_SUSPENDED, STATUS_HELD_SUSPENDED,
    STATUS_DATA_ERROR, STATUS_NO_VALUATION,
)


class Ledger:
    def __init__(self):
        self.rows = []      # 每期每只股票一行（持仓记录）
        self.trades = []    # 每笔交易一行（开仓 + 平仓）

    def record_period(self, signal_date, entry_date, exit_date,
                      long_symbols, short_symbols, weights,
                      prices_entry, prices_exit, future_rets, status_map):
        # ===== 1. 记录持仓行 =====
        for side, symbols in [("long", long_symbols), ("short", short_symbols)]:
            for sym in symbols:
                w = weights.get(sym, 0.0)
                status = status_map.get(sym, STATUS_FILLED)
                p0 = prices_entry.get(sym, np.nan)
                p1 = prices_exit.get(sym, np.nan)
                ret = future_rets.get(sym, np.nan)
                if status == STATUS_NO_VALUATION:
                    pnl = np.nan
                else:
                    pnl = w * ret if pd.notna(ret) else 0.0
                row = {
                    "signal_date": signal_date, "entry_date": entry_date,
                    "exit_date": exit_date, "symbol": sym, "side": side,
                    "target_weight": w, "price_entry": p0, "price_exit": p1,
                    "ret": ret, "status": status, "pnl_contrib": pnl,
                }
                self.rows.append(row)

                # ===== 2. 每条持仓生成一对开平仓交易 =====
                if status != STATUS_FILLED:
                    # 未成交 / 无估值 / 数据错误：不记交易
                    continue
                if pd.isna(p0) or pd.isna(p1):
                    continue

                # 开仓
                open_dir = "buy" if side == "long" else "sell"
                open_amount = abs(w)
                self.trades.append({
                    "signal_date": signal_date, "entry_date": entry_date,
                    "symbol": sym, "side": side, "direction": open_dir,
                    "trade_type": "open",
                    "price": p0, "amount": open_amount,
                    "commission": open_amount * COMMISSION_RATE,
                    "stamp_tax": open_amount * STAMP_TAX_RATE if open_dir == "sell" else 0.0,
                    "slippage": open_amount * SLIPPAGE_BPS / 10000.0,
                })

                # 平仓：金额按名义金额随价格变化
                close_dir = "sell" if side == "long" else "buy_to_cover"
                close_amount = abs(w) * (1.0 + ret) if pd.notna(ret) else abs(w)
                close_amount = max(close_amount, 0.0)   # 防止负值
                self.trades.append({
                    "signal_date": signal_date, "entry_date": exit_date,
                    "symbol": sym, "side": side, "direction": close_dir,
                    "trade_type": "close",
                    "price": p1, "amount": close_amount,
                    "commission": close_amount * COMMISSION_RATE,
                    "stamp_tax": close_amount * STAMP_TAX_RATE if close_dir == "sell" else 0.0,
                    "slippage": close_amount * SLIPPAGE_BPS / 10000.0,
                })

    def reconcile(self):
        if not self.rows:
            return pd.DataFrame()
        df = pd.DataFrame(self.rows)
        trades = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        out = []
        for sd, g in df.groupby("signal_date"):
            gross = g["pnl_contrib"].sum()
            t = trades[trades["signal_date"] == sd] if not trades.empty else pd.DataFrame()
            cost = (t["commission"].sum() + t["stamp_tax"].sum() + t["slippage"].sum()) if not t.empty else 0.0
            out.append({
                "signal_date": sd, "gross_pnl": gross, "total_cost": cost,
                "net_pnl": gross - cost,
                "n_positions": len(g),
                "n_no_valuation": (g["status"] == STATUS_NO_VALUATION).sum(),
                "n_unfilled": (g["status"] == STATUS_UNFILLED_SUSPENDED).sum(),
                "n_close_no_price": 0,   # 新模型下不再产生
            })
        return pd.DataFrame(out)

    def dump(self):
        pd.DataFrame(self.rows).to_csv(f"{LEDGER_DIR}/holdings.csv", index=False)
        pd.DataFrame(self.trades).to_csv(f"{LEDGER_DIR}/trades.csv", index=False)
        self.reconcile().to_csv(f"{LEDGER_DIR}/recon.csv", index=False)
        print(f"账本已保存: {LEDGER_DIR}")