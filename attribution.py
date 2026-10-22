# attribution.py
"""
2025Q3 归因。
- 多空分解、集中度、前十大贡献。
- 行业/动量/波动暴露（市值暂缺）。
- 候选池行业对照：并集，缺失填 0，带符号合计为 0。
- 风格分组盈亏：空头内部分组 + 候选池口径分组，双口径输出。
- 分组表列：loss_share（正数，合计 1）、weight_share（正数，合计 1）、
  excess_loss_share（带符号，合计 0）、portfolio_drag（实际拖累组合的百分点）。
- 逐股分组标签：输出 holdings_with_groups.csv，可与分组盈亏对账。
"""
import os
import pandas as pd
import numpy as np
from config import ATTRIBUTION_DIR, LEDGER_DIR


def _add_group_metrics(pnl_df, short_pnl):
    """统一计算 loss_share / weight_share / excess_loss_share / portfolio_drag。"""
    pnl_df["loss_share"] = pnl_df["pnl"] / pnl_df["pnl"].sum()
    pnl_df["weight_share"] = pnl_df["weight"] / pnl_df["weight"].sum()
    pnl_df["excess_loss_share"] = pnl_df["loss_share"] - pnl_df["weight_share"]
    pnl_df["portfolio_drag"] = pnl_df["loss_share"] * short_pnl
    return pnl_df


def attribute_period(signal_date, industry_map=None, mktcap_map=None,
                     momentum_map=None, vol_map=None, candidate_symbols=None):
    df = pd.read_csv(f"{LEDGER_DIR}/holdings.csv")
    df = df[df["signal_date"] == signal_date].copy()
    if df.empty:
        raise ValueError(f"账本中无 {signal_date} 记录")

    os.makedirs(ATTRIBUTION_DIR, exist_ok=True)

    # 1. 多空分解
    long_pnl = df[df["side"] == "long"]["pnl_contrib"].sum()
    short_pnl = df[df["side"] == "short"]["pnl_contrib"].sum()

    # 2. 前十大
    df["abs_pnl"] = df["pnl_contrib"].abs()
    top10 = df.nlargest(10, "abs_pnl")[
        ["symbol", "side", "target_weight", "ret", "pnl_contrib", "abs_pnl", "status"]
    ]

    # 3. 集中度
    total_abs = df["abs_pnl"].sum()
    gross_loss = abs(df["pnl_contrib"].sum())
    short_loss = abs(short_pnl)
    hhi = ((df["abs_pnl"] / total_abs) ** 2).sum() if total_abs > 0 else np.nan
    top10_abs = top10["abs_pnl"].sum()
    top10_share_gross = top10_abs / gross_loss if gross_loss > 0 else np.nan
    top10_share_short = top10_abs / short_loss if short_loss > 0 else np.nan

    # 4. 空头端分布
    short_ret = df[df["side"] == "short"]["ret"].dropna()
    short_stats = {
        "mean": short_ret.mean(), "median": short_ret.median(),
        "p10": short_ret.quantile(0.10), "p90": short_ret.quantile(0.90),
        "positive_ratio": (short_ret > 0).mean(), "n": len(short_ret),
    }

    # 5. 映射
    short_df = df[df["side"] == "short"].copy()
    if industry_map:
        df["industry"] = df["symbol"].map(industry_map)
        short_df["industry"] = short_df["symbol"].map(industry_map)
    if momentum_map:
        df["momentum"] = df["symbol"].map(momentum_map)
        short_df["momentum"] = short_df["symbol"].map(momentum_map)
    if vol_map:
        df["vol"] = df["symbol"].map(vol_map)
        short_df["vol"] = short_df["symbol"].map(vol_map)

    exposures = {}

    # 6. 行业
    if industry_map:
        short_ind_w = short_df.groupby("industry")["target_weight"].apply(lambda x: x.abs().sum())
        short_ind_w = short_ind_w / short_ind_w.sum()
        exposures["industry_short"] = short_ind_w.sort_values()

        # 候选池行业权重
        if candidate_symbols:
            pool_ind = pd.Series([industry_map.get(s) for s in candidate_symbols if industry_map.get(s)])
            pool_share = pool_ind.value_counts(normalize=True)
        else:
            pool_share = pd.Series(dtype=float)

        # 并集，缺失填 0
        all_ind = sorted(set(short_ind_w.index) | set(pool_share.index))
        ow_df = pd.DataFrame({
            "industry": all_ind,
            "short_weight": [short_ind_w.get(i, 0.0) for i in all_ind],
            "universe_weight": [pool_share.get(i, 0.0) for i in all_ind],
        })
        ow_df["overweight"] = ow_df["short_weight"] - ow_df["universe_weight"]
        ow_df = ow_df.sort_values("overweight")
        exposures["industry_overweight"] = ow_df.set_index("industry")

        # 行业分组盈亏 + 统一指标
        industry_pnl = short_df.groupby("industry").agg(
            weight=("target_weight", lambda x: x.abs().sum()),
            pnl=("pnl_contrib", "sum"),
            n=("symbol", "count"),
        ).sort_values("pnl")
        industry_pnl = _add_group_metrics(industry_pnl, short_pnl)
        exposures["industry_pnl"] = industry_pnl

    # 7. 动量
    if momentum_map:
        # 空头内部 5 组
        short_df["momentum_group_short"] = pd.qcut(short_df["momentum"], 5, duplicates="drop")
        exposures["momentum_short"] = short_df.groupby("momentum_group_short")["target_weight"].apply(
            lambda x: x.abs().sum()
        )
        momentum_pnl = short_df.groupby("momentum_group_short").agg(
            weight=("target_weight", lambda x: x.abs().sum()),
            pnl=("pnl_contrib", "sum"),
            n=("symbol", "count"),
        )
        momentum_pnl = _add_group_metrics(momentum_pnl, short_pnl)
        exposures["momentum_pnl"] = momentum_pnl

        # 候选池口径 5 组
        if candidate_symbols:
            pool_mom = pd.Series({s: momentum_map.get(s) for s in candidate_symbols if momentum_map.get(s) is not None})
            pool_mom_group = pd.qcut(pool_mom, 5, duplicates="drop")
            pool_group_map = pool_mom_group.to_dict()
            short_df["momentum_group_pool"] = short_df["symbol"].map(pool_group_map)
            momentum_pnl_pool = short_df.dropna(subset=["momentum_group_pool"]).groupby("momentum_group_pool").agg(
                weight=("target_weight", lambda x: x.abs().sum()),
                pnl=("pnl_contrib", "sum"),
                n=("symbol", "count"),
            )
            momentum_pnl_pool = _add_group_metrics(momentum_pnl_pool, short_pnl)
            exposures["momentum_pnl_pool"] = momentum_pnl_pool

    # 8. 波动
    if vol_map:
        short_df["vol_group_short"] = pd.qcut(short_df["vol"], 5, duplicates="drop")
        exposures["vol_short"] = short_df.groupby("vol_group_short")["target_weight"].apply(
            lambda x: x.abs().sum()
        )
        vol_pnl = short_df.groupby("vol_group_short").agg(
            weight=("target_weight", lambda x: x.abs().sum()),
            pnl=("pnl_contrib", "sum"),
            n=("symbol", "count"),
        )
        vol_pnl = _add_group_metrics(vol_pnl, short_pnl)
        exposures["vol_pnl"] = vol_pnl

        if candidate_symbols:
            pool_vol = pd.Series({s: vol_map.get(s) for s in candidate_symbols if vol_map.get(s) is not None})
            pool_vol_group = pd.qcut(pool_vol, 5, duplicates="drop")
            pool_vol_map = pool_vol_group.to_dict()
            short_df["vol_group_pool"] = short_df["symbol"].map(pool_vol_map)
            vol_pnl_pool = short_df.dropna(subset=["vol_group_pool"]).groupby("vol_group_pool").agg(
                weight=("target_weight", lambda x: x.abs().sum()),
                pnl=("pnl_contrib", "sum"),
                n=("symbol", "count"),
            )
            vol_pnl_pool = _add_group_metrics(vol_pnl_pool, short_pnl)
            exposures["vol_pnl_pool"] = vol_pnl_pool

    # 9. 逐股分组标签
    df_out = df.copy()
    if "vol" in df_out.columns:
        df_out["vol_group_short"] = short_df.set_index("symbol")["vol_group_short"].reindex(df_out["symbol"]).values
        if "vol_group_pool" in short_df.columns:
            df_out["vol_group_pool"] = short_df.set_index("symbol")["vol_group_pool"].reindex(df_out["symbol"]).values
    if "momentum" in df_out.columns:
        df_out["momentum_group_short"] = short_df.set_index("symbol")["momentum_group_short"].reindex(df_out["symbol"]).values
        if "momentum_group_pool" in short_df.columns:
            df_out["momentum_group_pool"] = short_df.set_index("symbol")["momentum_group_pool"].reindex(df_out["symbol"]).values

    # 10. 数据问题
    data_issues = df[df["status"].isin(["no_valuation", "data_error"])]

    # 11. 输出
    summary = pd.DataFrame([{
        "signal_date": signal_date, "long_pnl": long_pnl, "short_pnl": short_pnl,
        "total_pnl": long_pnl + short_pnl, "hhi": hhi,
        "top10_abs": top10_abs,
        "top10_share_gross": top10_share_gross,
        "top10_share_short": top10_share_short,
        **{f"short_{k}": v for k, v in short_stats.items()},
        "n_data_issues": len(data_issues),
    }])
    summary.to_csv(f"{ATTRIBUTION_DIR}/{signal_date}_summary.csv", index=False)
    top10.to_csv(f"{ATTRIBUTION_DIR}/{signal_date}_top10.csv", index=False)
    data_issues.to_csv(f"{ATTRIBUTION_DIR}/{signal_date}_data_issues.csv", index=False)
    df_out.to_csv(f"{ATTRIBUTION_DIR}/{signal_date}_holdings_with_groups.csv", index=False)
    for k, v in exposures.items():
        v.to_csv(f"{ATTRIBUTION_DIR}/{signal_date}_{k}.csv")
    print(f"归因已保存: {ATTRIBUTION_DIR}/{signal_date}_*.csv")
    return summary, top10