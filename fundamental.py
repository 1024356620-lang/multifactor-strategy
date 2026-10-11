# fundamental.py
import baostock as bs
import pandas as pd
import os
from config import DATA_DIR

FIN_CACHE = os.path.join(DATA_DIR, 'fundamentals.pkl')


def _fetch_financials(code, year, quarter):
    """拉取单只股票单季度财务数据（不再 login/logout）"""
    rs = bs.query_profit_data(code=code, year=year, quarter=quarter)
    rows = []
    while (rs.error_code == '0') & rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None
    return pd.DataFrame(rows, columns=rs.fields)


def load_fundamentals(klines, force_refresh=False):
    """
    加载所有股票的所有季度财务数据，构建 PIT 对齐的基本面因子表。
    只在开始时 login 一次，结束时 logout 一次。
    """
    if os.path.exists(FIN_CACHE) and not force_refresh:
        print(f"从缓存加载基本面数据: {FIN_CACHE}")
        return pd.read_pickle(FIN_CACHE)

    codes = list(klines.keys())
    years = range(2019, 2026)
    quarters = [1, 2, 3, 4]

    print("登录 baostock...")
    bs.login()

    all_rows = []
    total = len(codes) * len(years) * len(quarters)
    done = 0

    for i, code in enumerate(codes):
        for year in years:
            for q in quarters:
                try:
                    df = _fetch_financials(code, year, q)
                    if df is not None and not df.empty:
                        all_rows.append(df)
                except Exception as e:
                    print(f"  {code} {year}Q{q} 失败: {e}")
                done += 1
        print(f"  财务数据已处理 {i+1}/{len(codes)} 只（总请求 {done}/{total}）")

    bs.logout()

    if not all_rows:
        print("警告: 未获取到任何财务数据")
        return pd.DataFrame()

    fin = pd.concat(all_rows, ignore_index=True)
    fin['pubDate'] = pd.to_datetime(fin['pubDate'])
    fin['statDate'] = pd.to_datetime(fin['statDate'])

    for col in ['roeAvg', 'npMargin', 'epsTTM', 'MBRevenue', 'totalShare', 'liqaShare']:
        if col in fin.columns:
            fin[col] = pd.to_numeric(fin[col], errors='coerce')

    fin = fin.sort_values(['code', 'pubDate']).reset_index(drop=True)
    pd.to_pickle(fin, FIN_CACHE)
    print(f"基本面数据已缓存: {FIN_CACHE}")
    return fin