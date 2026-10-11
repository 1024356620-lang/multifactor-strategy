# main.py
import os
import pandas as pd
import numpy as np
from config import MIN_STOCKS_PER_PERIOD, TRANSACTION_COST, FACTOR_COLS, FACTOR_WEIGHTS
from data_loader import load_all_data
from factor_calc import build_cross_section
from preprocess import preprocess_factors
from ic_analysis import calc_ic, summarize_ic
from backtest import calc_period_return, composite_score, performance_summary
from fundamental import load_fundamentals
from valuation import calc_valuation_factors

# 确保 results 目录存在
os.makedirs('results', exist_ok=True)


def layered_returns(factor_df, future_rets, n_groups=5):
    df = factor_df.copy()
    df['score'] = composite_score(df)
    df['future_ret'] = df['symbol'].map(future_rets)
    df = df.dropna(subset=['future_ret']).sort_values('score', ascending=False)
    if len(df) < n_groups * 5:
        return None
    df['group'] = pd.qcut(
        df['score'].rank(method='first'),
        n_groups,
        labels=[f'G{i+1}' for i in range(n_groups)]
    )
    return df.groupby('group')['future_ret'].mean()


def main():
    # ========== 1. 数据 ==========
    klines = load_all_data(force_refresh=False)
    print(f"股票数: {len(klines)}")

    # ========== 2. 基本面数据 ==========
    print("\n加载基本面数据...")
    try:
        fin_df = load_fundamentals(klines, force_refresh=False)
    except Exception as e:
        print(f"基本面加载失败: {e}，将只使用量价因子")
        fin_df = pd.DataFrame()

    if fin_df.empty:
        print("警告: 基本面数据为空")
    else:
        print(f"基本面数据行数: {len(fin_df)}")

    # ========== 3. 生成调仓日（季度）==========
    all_dates = sorted(set().union(*[set(df['date']) for df in klines.values()]))
    all_dates = pd.to_datetime(all_dates)
    quarter_groups = pd.Series(all_dates).dt.to_period('Q')
    rebalance_dates = [
        all_dates[quarter_groups[quarter_groups == p].index[0]]
        for p in quarter_groups.drop_duplicates()
    ]
    rebalance_dates = [d for d in rebalance_dates if d < all_dates[-1]]
    print(f"调仓次数: {len(rebalance_dates)}")

    # ========== 4. 逐期回测 ==========
    ic_records = []
    ls_records = []
    layer_records = []
    nav_gross = [1.0]
    nav_net = [1.0]

    first_valid_done = False

    for i, rb_date in enumerate(rebalance_dates):
        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        if next_date is None:
            break

        # 4.1 量价因子
        factor_df = build_cross_section(klines, rb_date)
        if factor_df is None or len(factor_df) < MIN_STOCKS_PER_PERIOD:
            print(f"{rb_date.date()} 有效样本不足，跳过")
            continue

        # 4.2 合并 EP
        if not fin_df.empty:
            val_df = calc_valuation_factors(klines, fin_df, rb_date)
            if not val_df.empty:
                factor_df = factor_df.merge(val_df, on='symbol', how='left')
            else:
                factor_df['EP'] = 0.0
        else:
            factor_df['EP'] = 0.0

        if 'EP' not in factor_df.columns:
            factor_df['EP'] = 0.0
        factor_df['EP'] = factor_df['EP'].fillna(factor_df['EP'].median())

        # 4.3 预处理
        factor_processed = preprocess_factors(factor_df)
        if factor_processed is None or factor_processed.empty:
            print(f"{rb_date.date()} 预处理后为空，跳过")
            continue

        # 4.4 下期收益
        future_rets = calc_period_return(klines, rb_date, next_date)

        # 4.5 IC 计算
        temp = factor_processed.copy()
        temp['future_ret'] = temp['symbol'].map(future_rets)
        valid = temp.dropna(subset=['future_ret'])
        if len(valid) < MIN_STOCKS_PER_PERIOD:
            print(f"{rb_date.date()} 收益样本不足，跳过")
            continue

        if not first_valid_done:
            first_valid_done = True
            print(f"\n第一期有效 @ {rb_date.date()}, 样本数 {len(valid)}")

        ic_dict = calc_ic(valid)
        ic_dict['date'] = rb_date
        ic_records.append(ic_dict)

        # 4.6 多空组合（前25% vs 后25%，等权）
        scored = factor_processed.copy()
        scored['score'] = composite_score(scored)
        scored['future_ret'] = scored['symbol'].map(future_rets)
        scored = scored.dropna(subset=['future_ret']).sort_values(
            'score', ascending=False
        )
        n = max(1, int(len(scored) * 0.25))
        long_ret = scored.head(n)['future_ret'].mean()
        short_ret = scored.tail(n)['future_ret'].mean()
        ls_gross = long_ret - short_ret
        ls_net = ls_gross - 2 * TRANSACTION_COST

        ls_records.append({
            'date': rb_date, 'long': long_ret, 'short': short_ret,
            'ls_gross': ls_gross, 'ls_net': ls_net,
        })
        nav_gross.append(nav_gross[-1] * (1 + ls_gross))
        nav_net.append(nav_net[-1] * (1 + ls_net))

        # 4.7 分层回测
        layer = layered_returns(factor_processed, future_rets, n_groups=5)
        if layer is not None:
            layer_dict = layer.to_dict()
            layer_dict['date'] = rb_date
            layer_records.append(layer_dict)

        if (i + 1) % 10 == 0:
            print(f"  已完成 {i+1}/{len(rebalance_dates)} 期")

    # ========== 5. 汇总 ==========
    print("\n" + "=" * 50)
    print("因子 IC 汇总")
    print("=" * 50)
    ic_summary = summarize_ic(ic_records)
    if not ic_summary.empty:
        print(ic_summary.to_string(index=False))
        ic_summary.to_csv('results/ic_summary.csv', index=False)

    print("\n" + "=" * 50)
    print("分层回测（G1=打分最低，G5=打分最高）")
    print("=" * 50)
    if layer_records:
        layer_df = pd.DataFrame(layer_records).set_index('date')
        layer_mean = layer_df.mean()
        print(layer_mean.to_string())
        print(f"\n  G5 - G1: {layer_mean.get('G5', 0) - layer_mean.get('G1', 0):.4%}")
        g_cols = [f'G{i+1}' for i in range(5)]
        vals = [layer_mean.get(c, np.nan) for c in g_cols]
        if all(pd.notna(v) for v in vals):
            if all(vals[j] <= vals[j+1] for j in range(4)):
                print("  单调性: 单调递增 ✓")
            elif all(vals[j] >= vals[j+1] for j in range(4)):
                print("  单调性: 单调递减 ✗")
            else:
                print("  单调性: 非单调")
        layer_df.to_csv('results/layered_returns.csv')

    print("\n" + "=" * 50)
    print("多空组合绩效")
    print("=" * 50)
    if ls_records:
        ls_df = pd.DataFrame(ls_records).set_index('date')
        print("\n【毛收益】")
        for k, v in performance_summary(nav_gross, ls_df['ls_gross']).items():
            print(f"  {k}: {v}")
        print("\n【净收益】")
        for k, v in performance_summary(nav_net, ls_df['ls_net']).items():
            print(f"  {k}: {v}")
        print(f"\n成本侵蚀: 毛 {nav_gross[-1]-1:.2%} → 净 {nav_net[-1]-1:.2%}")
        ls_df.to_csv('results/long_short.csv')
        pd.Series(nav_gross, name='nav_gross').to_csv('results/nav_gross.csv', index=False)
        pd.Series(nav_net, name='nav_net').to_csv('results/nav_net.csv', index=False)
        print("\n结果已保存到 results/")


if __name__ == '__main__':
    main()