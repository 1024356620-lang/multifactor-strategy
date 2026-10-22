# daily_nav.py
"""
逐日盯市净值（权重漂移版）。
- 每期 entry_date 设定目标权重。
- 之后每天，用上一日的权重 × 当日个股收益，更新组合收益，权重自然漂移。
- 空头权重为负，收益贡献为负。
- 不扣费用（日频净值是毛收益）。
"""
import os
import pandas as pd
import numpy as np
from config import DAILY_NAV_DIR

os.makedirs(DAILY_NAV_DIR, exist_ok=True)


def build_price_panel(klines, start, end):
    frames = []
    for sym, df in klines.items():
        s = df.set_index("date")["close"].rename(sym)
        frames.append(s)
    panel = pd.concat(frames, axis=1).sort_index()
    panel = panel.loc[(panel.index >= pd.to_datetime(start)) & (panel.index <= pd.to_datetime(end))]
    panel = panel.where(panel > 1e-6)
    return panel

def daily_metrics(nav, periods_per_year=252):
    """日频指标：算术年化、年化波动、算术夏普、最大回撤。"""
    ret = nav.pct_change().dropna()
    if len(ret) == 0:
        return {}
    arithmetic_ann = ret.mean() * periods_per_year
    ann_vol = ret.std() * np.sqrt(periods_per_year)
    sharpe_arith = arithmetic_ann / ann_vol if ann_vol > 0 else np.nan
    dd = (nav / nav.cummax() - 1).min()
    return {
        "daily_ann_ret": arithmetic_ann,   # 算术年化
        "daily_ann_vol": ann_vol,
        "daily_sharpe": sharpe_arith,      # 算术夏普
        "daily_max_dd": dd,
    }

def compute_daily_nav(price_panel, holdings, signal_dates, entry_dates, exit_dates):
    dates = price_panel.index
    nav = pd.Series(index=dates, dtype=float)
    nav.iloc[0] = 1.0

    # 每期权重
    period_weights = []
    for sd, ed, xd in zip(signal_dates, entry_dates, exit_dates):
        h = holdings[holdings["signal_date"] == sd]
        if h.empty:
            continue
        w = h.set_index("symbol")["target_weight"].to_dict()
        ed = pd.to_datetime(ed)
        valid = dates[dates >= ed]
        if len(valid) == 0:
            continue
        period_weights.append({"entry": valid[0], "weights": w})

    # 当前权重（会漂移）
    current_w = {}
    prev_nav = 1.0

    for i in range(1, len(dates)):
        d = dates[i]
        prev_d = dates[i - 1]

        # entry_date：切换权重
        for pw in period_weights:
            if pw["entry"] == d:
                current_w = dict(pw["weights"])
                break

        if not current_w:
            nav.iloc[i] = prev_nav
            continue

        # 逐日收益：用 prev_d 的权重 × 当日个股收益
        port_ret = 0.0
        new_w = {}
        total_abs = 0.0
        for sym, w in current_w.items():
            if sym not in price_panel.columns:
                new_w[sym] = w
                total_abs += abs(w)
                continue
            p0 = price_panel.at[prev_d, sym] if prev_d in price_panel.index else np.nan
            p1 = price_panel.at[d, sym] if d in price_panel.index else np.nan
            if pd.notna(p0) and pd.notna(p1) and p0 > 1e-6 and p1 > 1e-6:
                r = p1 / p0 - 1
                port_ret += w * r
                # 权重漂移：w_new = w * (1 + r) / (1 + port_ret)
                new_w[sym] = w * (1 + r)
            else:
                # 停牌：权重不变，收益按 0
                new_w[sym] = w
        # 归一化权重，使绝对值之和保持为初始值
        init_abs = sum(abs(v) for v in current_w.values())
        if init_abs > 0:
            scale = init_abs / sum(abs(v) for v in new_w.values()) if sum(abs(v) for v in new_w.values()) > 0 else 1.0
            new_w = {k: v * scale for k, v in new_w.items()}

        prev_nav = nav.iloc[i - 1] * (1 + port_ret)
        nav.iloc[i] = prev_nav
        current_w = new_w

    nav = nav.ffill()
    nav.iloc[0] = 1.0
    return nav