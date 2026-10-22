# run_frozen_test.py
"""
冻结 50/50 策略，一次性运行 2025Q4 - 2026Q3 完整 4 个季度。
输出：配置、逐期账本、持仓明细。
不调参，不扫描。
"""
import os
import json
import pandas as pd
import numpy as np

from data_loader import load_all_data
from factor_calc import build_cross_section
from preprocess import preprocess_factors
from ledger import Ledger

EXP_DIR = "experiments_frozen_2025Q4_2026Q3"
os.makedirs(EXP_DIR, exist_ok=True)

# 冻结参数
CONFIG = {
    "start_date": "2025-10-01",
    "end_date": "2026-09-30",
    "lookback_window": 60,
    "top_pct": 0.25,
    "weight_combo": 0.5,
    "cost_rate": 0.0025,
}

def build_intervals(all_dates, start, end):
    dates = all_dates[(all_dates >= pd.to_datetime(start)) & (all_dates <= pd.to_datetime(end))]
    qg = pd.Series(dates).dt.to_period("Q")
    signal_dates = [dates[qg[qg == p].index[0]] for p in qg.drop_duplicates()]
    signal_dates = [d for d in signal_dates if d < dates[-1]]
    intervals = []
    for i, sig in enumerate(signal_dates):
        nxt = dates[-1] if i == len(signal_dates) - 1 else signal_dates[i + 1]
        future = dates[dates > sig]
        if len(future) == 0: continue
        entry = future[0]
        intervals.append({"signal_date": sig, "entry_date": entry, "exit_date": nxt})
    return intervals

def calc_lowvol_at_date(df, end_date, window=60):
    sub = df[df["date"] <= pd.to_datetime(end_date)].tail(window + 1)
    if len(sub) < window: return np.nan
    rets = sub["close"].pct_change().dropna()
    if len(rets) < 10: return np.nan
    return rets.std() * np.sqrt(252)

def run_frozen(klines, intervals):
    lgs = {
        "market_equal": Ledger(),
        "lowvol_only": Ledger(),
        "combo_50_50": Ledger()
    }

    for it in intervals:
        rb, entry, exit_ = it["signal_date"], it["entry_date"], it["exit_date"]
        factor_df = build_cross_section(klines, rb)
        if factor_df is None or len(factor_df) < 20: continue
        if "EP" not in factor_df.columns: factor_df["EP"] = 0.0

        lowvol_vals = {}
        for sym, df in klines.items():
            v = calc_lowvol_at_date(df, rb, window=CONFIG["lookback_window"])
            if pd.notna(v): lowvol_vals[sym] = v
        factor_df["lowvol"] = factor_df["symbol"].map(lowvol_vals)

        factor_processed = preprocess_factors(factor_df)
        if factor_processed is None or factor_processed.empty: continue

        if "lowvol" in factor_processed.columns:
            factor_processed["lowvol"] = -factor_processed["lowvol"]

        # 1. market_equal
        syms_market = factor_processed["symbol"].tolist()
        weights_market = {s: 1.0 / len(syms_market) for s in syms_market}

        # 2. lowvol_only
        scored_lv = factor_processed.sort_values("lowvol", ascending=False)
        n_lv = max(1, int(len(scored_lv) * CONFIG["top_pct"]))
        syms_lv = scored_lv.head(n_lv)["symbol"].tolist()
        weights_lv = {s: 1.0 / n_lv for s in syms_lv}

        # 3. combo_50_50 (两因子 50% + 低波动 50%)
        scored_2f = factor_processed.copy()
        scored_2f["score"] = 0.5 * scored_2f["reversal"] + 0.5 * scored_2f["ma_ratio"]
        scored_2f = scored_2f.sort_values("score", ascending=False)
        n_2f = max(1, int(len(scored_2f) * CONFIG["top_pct"]))
        syms_2f = scored_2f.head(n_2f)["symbol"].tolist()
        
        weights_combo = {}
        for s in syms_2f: weights_combo[s] = weights_combo.get(s, 0.0) + CONFIG["weight_combo"] / n_2f
        for s in syms_lv: weights_combo[s] = weights_combo.get(s, 0.0) + CONFIG["weight_combo"] / n_lv
        syms_combo = list(weights_combo.keys())

        # 记录
        def process_record(name, lg, weights, syms):
            pe, px, st, fr = {}, {}, {}, {}
            for sym in syms:
                df = klines.get(sym)
                if df is None:
                    st[sym] = "data_error"; continue
                er = df[df["date"] == entry]
                er = er.iloc[0] if not er.empty else None
                es = df[(df["date"] >= entry) & (df["date"] <= exit_)]
                vc = es["close"].dropna()
                vc = vc[vc > 0]
                p0 = er["open"] if er is not None and pd.notna(er["open"]) and er["open"] > 0 else np.nan
                p1 = vc.iloc[-1] if len(vc) else np.nan
                pe[sym] = p0; px[sym] = p1
                if pd.isna(p0):
                    st[sym] = "unfilled_suspended"; fr[sym] = np.nan
                elif pd.isna(p1):
                    st[sym] = "no_valuation"; fr[sym] = np.nan
                else:
                    st[sym] = "filled"; fr[sym] = p1 / p0 - 1
            lg.record_period(rb, entry, exit_, syms, [], weights, pe, px, fr, st)
            
            # 输出当期持仓明细
            pd.DataFrame(lg.rows).to_csv(f"{EXP_DIR}/{name}_holdings_{rb.date()}.csv", index=False)

        process_record("market_equal", lgs["market_equal"], weights_market, syms_market)
        process_record("lowvol_only", lgs["lowvol_only"], weights_lv, syms_lv)
        process_record("combo_50_50", lgs["combo_50_50"], weights_combo, syms_combo)

    return lgs

def main():
    # 保存配置
    with open(f"{EXP_DIR}/frozen_config.json", "w") as f:
        json.dump(CONFIG, f, indent=4)
    
    klines = load_all_data(force_refresh=False)
    all_dates = pd.to_datetime(sorted(set().union(*[set(df["date"]) for df in klines.values()])))
    
    intervals = build_intervals(all_dates, CONFIG["start_date"], CONFIG["end_date"])
    print(f"冻结测试调仓期数: {len(intervals)}")
    if len(intervals) != 4:
        print("警告：期数不为 4，请检查日期范围和数据覆盖情况！")

    lgs = run_frozen(klines, intervals)

    summary = []
    for name, lg in lgs.items():
        recon = lg.reconcile().sort_values("signal_date").reset_index(drop=True)
        recon.to_csv(f"{EXP_DIR}/{name}_recon.csv", index=False)
        
        net = recon["net_pnl"].dropna()
        cum_ret = (1 + net).prod() - 1
        sharpe_q = net.mean() / net.std(ddof=1) * np.sqrt(4) if net.std(ddof=1) > 0 else np.nan
        nav = (1 + net).cumprod()
        nav_full = np.r_[1.0, nav.values]
        max_dd = (nav_full / np.maximum.accumulate(nav_full) - 1).min()
        turnover = recon["total_cost"].sum() / CONFIG["cost_rate"] / len(recon) if len(recon) > 0 else np.nan

        summary.append({
            "scheme": name, "n_periods": len(net), 
            "net_cum": cum_ret, "sharpe_q": sharpe_q, 
            "max_dd": max_dd, "avg_turnover": turnover
        })
    
    df = pd.DataFrame(summary)
    df.to_csv(f"{EXP_DIR}/frozen_summary.csv", index=False)
    print("\n=== 冻结策略一次性验证 (2025Q4 - 2026Q3) ===")
    print(df.to_string(index=False))
    print(f"\n所有配置、逐期账本及持仓明细已保存至: {EXP_DIR}/")

if __name__ == "__main__":
    main()