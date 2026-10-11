# diagnose.py
"""
独立诊断脚本：打印指定调仓日的 G5 组构成。
用法：python diagnose.py
"""
import pandas as pd
import numpy as np
from config import MIN_STOCKS_PER_PERIOD
from data_loader import load_all_data
from factor_calc import build_cross_section
from preprocess import preprocess_factors
from backtest import calc_period_return, composite_score


def main():
    # ========== 1. 加载数据 ==========
    klines = load_all_data(force_refresh=False)
    print(f"股票数: {len(klines)}")

    # ========== 2. 生成调仓日 ==========
    all_dates = sorted(set().union(*[set(df['date']) for df in klines.values()]))
    all_dates = pd.to_datetime(all_dates)
    quarter_groups = pd.Series(all_dates).dt.to_period('Q')
    rebalance_dates = [
        all_dates[quarter_groups[quarter_groups == p].index[0]]
        for p in quarter_groups.drop_duplicates()
    ]
    rebalance_dates = [d for d in rebalance_dates if d < all_dates[-1]]
    print(f"调仓次数: {len(rebalance_dates)}")

    # ========== 3. 找一个有效调仓日 ==========
    target_date = None
    factor_df = None
    for rb_date in rebalance_dates:
        fdf = build_cross_section(klines, rb_date)
        if fdf is not None and len(fdf) >= MIN_STOCKS_PER_PERIOD:
            target_date = rb_date
            factor_df = fdf
            break

    if target_date is None:
        print("找不到有效调仓日")
        return

    print(f"\n使用调仓日: {target_date.date()}")
    print(f"原始因子截面形状: {factor_df.shape}")

    # ========== 4. 合并 EP 因子 ==========
    try:
        from fundamental import load_fundamentals
        from valuation import calc_valuation_factors
        fin_df = load_fundamentals(klines, force_refresh=False)
        if not fin_df.empty:
            val_df = calc_valuation_factors(klines, fin_df, target_date)
            if not val_df.empty:
                factor_df = factor_df.merge(val_df, on='symbol', how='left')
            else:
                factor_df['EP'] = 0.0
        else:
            factor_df['EP'] = 0.0
    except Exception as e:
        print(f"基本面加载失败: {e}，EP 设为 0")
        factor_df['EP'] = 0.0

    if 'EP' not in factor_df.columns:
        factor_df['EP'] = 0.0
    factor_df['EP'] = factor_df['EP'].fillna(factor_df['EP'].median())

    # ========== 5. 预处理 ==========
    factor_processed = preprocess_factors(factor_df)
    if factor_processed is None or factor_processed.empty:
        print("预处理后为空")
        return

    print(f"预处理后形状: {factor_processed.shape}")
    print(f"列: {factor_processed.columns.tolist()}")

    # ========== 6. 计算下期收益 ==========
    idx = rebalance_dates.index(target_date)
    if idx + 1 >= len(rebalance_dates):
        print("没有下一期")
        return
    next_date = rebalance_dates[idx + 1]
    future_rets = calc_period_return(klines, target_date, next_date)
    print(f"下期收益样本数: {len(future_rets)}")

    # ========== 7. 合成打分 ==========
    scored = factor_processed.copy()
    scored['score'] = composite_score(scored)
    scored['future_ret'] = scored['symbol'].map(future_rets)
    scored = scored.dropna(subset=['future_ret']).sort_values('score', ascending=False)

    print(f"有效打分样本数: {len(scored)}")

    # ========== 8. 打印 G5 构成 ==========
    n_g5 = max(1, len(scored) // 5)
    g5 = scored.head(n_g5)

    print("\n" + "=" * 70)
    print(f"G5 组构成（打分最高 {n_g5} 只）")
    print("=" * 70)

    cols_show = ['symbol', 'momentum', 'volatility', 'avg_turn',
                 'ma_ratio', 'EP', 'score', 'future_ret']
    cols_show = [c for c in cols_show if c in g5.columns]
    print(g5[cols_show].to_string(index=False))

    print(f"\nG5 平均未来收益: {g5['future_ret'].mean():.4%}")
    print(f"全样本平均未来收益: {scored['future_ret'].mean():.4%}")

    print(f"\nG5 各因子均值 vs 全样本均值:")
    print(f"{'因子':<12} {'G5均值':>10} {'全样本':>10} {'差异':>10}")
    for c in ['momentum', 'volatility', 'avg_turn', 'ma_ratio', 'EP']:
        if c in g5.columns:
            g5_mean = g5[c].mean()
            all_mean = scored[c].mean()
            print(f"{c:<12} {g5_mean:>+10.4f} {all_mean:>+10.4f} {g5_mean - all_mean:>+10.4f}")

    # ========== 9. 代码前缀分布 ==========
    print(f"\nG5 代码前缀分布:")
    prefixes = g5['symbol'].str[:6].value_counts()
    print(prefixes.to_string())

    print(f"\n全样本代码前缀分布:")
    all_prefixes = scored['symbol'].str[:6].value_counts()
    print(all_prefixes.head(10).to_string())

    # ========== 10. 对比 G1 ==========
    g1 = scored.tail(n_g5)
    print("\n" + "=" * 70)
    print(f"G1 组构成（打分最低 {n_g5} 只）")
    print("=" * 70)
    print(g1[cols_show].to_string(index=False))
    print(f"\nG1 平均未来收益: {g1['future_ret'].mean():.4%}")


if __name__ == '__main__':
    main()