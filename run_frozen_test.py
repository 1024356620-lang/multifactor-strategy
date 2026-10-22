# run_frozen_test.py
import os, json
import pandas as pd
import numpy as np
from data_loader import load_all_data
from factor_calc import build_cross_section
from preprocess import preprocess_factors
from ledger import Ledger
import config

EXP_DIR = "experiments_frozen_2025Q4_2026Q3"
os.makedirs(EXP_DIR, exist_ok=True)

CONFIG = {
    "start_date": "2025-10-01",
    "end_date": "2026-09-30",
    "lookback_window": 60,
    "top_pct": 0.25,
    "weight_combo": 0.5,
    "cost_rate": 0.0025
}

def build_intervals(all_dates, start, end):
    dates = all_dates[(all_dates >= pd.to_datetime(start)) & (all_dates <= pd.to_datetime(end))]
    qg = pd.Series(dates).dt.to_period("Q")
    intervals = []
    for period, group in qg.groupby(qg):
        group_dates = dates[qg == period]
        if len(group_dates) < 1:
            continue
        sig_date = group_dates[0]
        future_dates = dates[dates > sig_date]
        if len(future_dates) == 0:
            continue
        entry_date = future_dates[0]
        exit_date = group_dates[-1]
        if entry_date > exit_date: 
            entry_date = exit_date
        intervals.append({"signal_date": sig_date, "entry_date": entry_date, "exit_date": exit_date})
    return intervals

def calc_lowvol(df, end_date, window=60):
    sub = df[df["date"] <= pd.to_datetime(end_date)].tail(window + 1)
    if len(sub) < window + 1: 
        return 99.0
    rets = sub["close"].pct_change().dropna()
    if len(rets) < window: 
        return 99.0
    return rets.std() * np.sqrt(252)

def main():
    with open(f"{EXP_DIR}/frozen_config.json", "w") as f:
        json.dump(CONFIG, f, indent=4)
    
    config.COMMISSION_RATE = CONFIG["cost_rate"]
    config.STAMP_TAX_RATE = 0.0
    config.SLIPPAGE_BPS = 0

    klines = load_all_data(force_refresh=False)
    all_dates = pd.to_datetime(sorted(set().union(*[set(df["date"]) for df in klines.values()])))
    intervals = build_intervals(all_dates, CONFIG["start_date"], CONFIG["end_date"])
    
    print(f"=== 冻结测试调仓期数: {len(intervals)} ===")

    lgs = {
        "market_equal": Ledger(commission_rate=CONFIG["cost_rate"]),
        "lowvol_only": Ledger(commission_rate=CONFIG["cost_rate"]),
        "combo_50_50": Ledger(commission_rate=CONFIG["cost_rate"])
    }

    for it in intervals:
        rb, entry, exit_ = it["signal_date"], it["entry_date"], it["exit_date"]
        f_df = build_cross_section(klines, rb)
        if f_df is None or len(f_df) < 20: 
            for lg in lgs.values():
                lg.record_period(rb, entry, exit_, [], [], {}, {}, {}, {}, {})
            continue
            
        if "EP" not in f_df.columns: f_df["EP"] = 0.0
        
        lowvol = {}
        for s, df in klines.items():
            v = calc_lowvol(df, rb, CONFIG["lookback_window"])
            if pd.notna(v): lowvol[s] = v
        f_df["lowvol"] = f_df["symbol"].map(lowvol)
        
        f_proc = preprocess_factors(f_df)
        if f_proc is None or f_proc.empty: 
            for lg in lgs.values():
                lg.record_period(rb, entry, exit_, [], [], {}, {}, {}, {}, {})
            continue

        f_proc["lowvol"] = f_proc["symbol"].map(lowvol).fillna(99.0)
        f_proc["lowvol"] = f_proc["lowvol"].rank(pct=True) - 0.5
        f_proc["lowvol"] = -f_proc["lowvol"]

        syms_m = f_proc["symbol"].tolist()
        w_m = {s: 1.0/len(syms_m) for s in syms_m}

        scored_lv = f_proc.sort_values("lowvol", ascending=False)
        n_lv = max(1, int(len(scored_lv) * CONFIG["top_pct"]))
        syms_lv = scored_lv.head(n_lv)["symbol"].tolist()
        w_lv = {s: 1.0/n_lv for s in syms_lv}

        scored_2f = f_proc.copy()
        scored_2f["score"] = 0.5 * scored_2f["reversal"] + 0.5 * scored_2f["ma_ratio"]
        scored_2f = scored_2f.sort_values("score", ascending=False)
        n_2f = max(1, int(len(scored_2f) * CONFIG["top_pct"]))
        syms_2f = scored_2f.head(n_2f)["symbol"].tolist()
        
        w_cb = {}
        w_combo = CONFIG["weight_combo"]
        for s in syms_2f: w_cb[s] = w_cb.get(s, 0.0) + w_combo / n_2f
        for s in syms_lv: w_cb[s] = w_cb.get(s, 0.0) + (1 - w_combo) / n_lv
        syms_cb = list(w_cb.keys())

        def rec(name, lg, weights, syms):
            pe, px, st, fr = {}, {}, {}, {}
            for sym in syms:
                df = klines.get(sym)
                if df is None: 
                    st[sym] = "data_error"; 
                    continue
                er = df[df["date"] == entry].iloc[0] if not df[df["date"] == entry].empty else None
                es = df[(df["date"] >= entry) & (df["date"] <= exit_)]
                vc = es["close"].dropna()
                vc = vc[vc > 0]
                
                p0 = er["open"] if er is not None and pd.notna(er["open"]) and er["open"] > 0 else np.nan
                p1 = vc.iloc[-1] if len(vc) else np.nan
                pe[sym], px[sym] = p0, p1
                
                if pd.isna(p0): 
                    st[sym] = "unfilled_suspended"; fr[sym] = np.nan
                elif pd.isna(p1): 
                    st[sym] = "no_valuation"; fr[sym] = np.nan
                else: 
                    st[sym] = "filled"; fr[sym] = p1 / p0 - 1
            lg.record_period(rb, entry, exit_, syms, [], weights, pe, px, fr, st)
            # 保存持仓明细
            pd.DataFrame(lg.rows).to_csv(f"{EXP_DIR}/{name}_holdings_{rb.date()}.csv", index=False)

        rec("market_equal", lgs["market_equal"], w_m, syms_m)
        rec("lowvol_only", lgs["lowvol_only"], w_lv, syms_lv)
        rec("combo_50_50", lgs["combo_50_50"], w_cb, syms_cb)

    summary = []
    for name, lg in lgs.items():
        recon = lg.reconcile().sort_values("signal_date").reset_index(drop=True)
        recon.to_csv(f"{EXP_DIR}/{name}_recon.csv", index=False)
        net = recon["net_pnl"].dropna()
        cum_ret = (1 + net).prod() - 1
        sharpe = net.mean() / net.std(ddof=1) * np.sqrt(4) if net.std(ddof=1) > 0 else np.nan
        nav = np.r_[1.0, (1 + net).cumprod().values]
        max_dd = (nav / np.maximum.accumulate(nav) - 1).min()
        turnover = recon["total_cost"].sum() / CONFIG["cost_rate"] / len(recon) if len(recon) > 0 else np.nan
        summary.append({"scheme": name, "n_periods": len(net), "net_cum": cum_ret, "sharpe_q": sharpe, "max_dd": max_dd, "avg_turnover": turnover})

    df = pd.DataFrame(summary)
    df.to_csv(f"{EXP_DIR}/frozen_summary.csv", index=False)
    print("\n=== 冻结策略一次性验证 ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()