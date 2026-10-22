# run_attribution.py
import pandas as pd
from attribution import attribute_period
from data_loader import load_all_data
from factor_calc import calc_factors_at_date, build_cross_section
from industry import load_industry

SIGNAL_DATE = "2025-07-01"

klines = load_all_data(force_refresh=False)
ind_df = load_industry(force_refresh=False)
industry_map = dict(zip(ind_df["symbol"], ind_df["industry"]))

# 动量 / 波动映射
momentum_map, vol_map = {}, {}
for sym, df in klines.items():
    f = calc_factors_at_date(df, SIGNAL_DATE)
    if f:
        momentum_map[sym] = f["ma_ratio"]
        vol_map[sym] = f["volatility"]

# 候选池 = 当期 build_cross_section 返回的全部股票（不经过 preprocess_factors，避免 EP 缺列）
factor_df = build_cross_section(klines, SIGNAL_DATE)
candidate_symbols = set(factor_df["symbol"]) if factor_df is not None else set()
print(f"候选池股票数: {len(candidate_symbols)}")

summary, top10 = attribute_period(
    SIGNAL_DATE,
    industry_map=industry_map,
    momentum_map=momentum_map,
    vol_map=vol_map,
    candidate_symbols=candidate_symbols,
)

print("\n归因汇总：")
print(summary.to_string(index=False))
print("\n前十大贡献：")
print(top10.to_string(index=False))

# 校验行业超配合计为 0
ow = pd.read_csv(f"output/attribution/{SIGNAL_DATE}_industry_overweight.csv")
print(f"\n行业超配合计: {ow['overweight'].sum():.6f}（应 ≈ 0）")
print(f"空头权重合计: {ow['short_weight'].sum():.6f}（应 = 1）")
print(f"候选池权重合计: {ow['universe_weight'].sum():.6f}（应 = 1）")