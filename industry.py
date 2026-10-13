# industry.py
import baostock as bs
import pandas as pd
import os
from config import DATA_DIR

INDUSTRY_CACHE = os.path.join(DATA_DIR, 'industry.pkl')


# industry.py
def fetch_industry():
    print("1. 准备登录 baostock...")
    lg = bs.login()
    print(f"   登录返回: error_code={lg.error_code}, msg={lg.error_msg}")

    print("2. 开始查询行业分类...")
    rs = bs.query_stock_industry()
    print(f"   查询返回: error_code={rs.error_code}")

    rows = []
    count = 0
    while (rs.error_code == '0') & rs.next():
        rows.append(rs.get_row_data())
        count += 1
        if count % 500 == 0:
            print(f"   已拉取 {count} 条")

    print(f"3. 拉取完成，共 {len(rows)} 条")
    bs.logout()
    print("4. 已登出")

    df = pd.DataFrame(rows, columns=rs.fields)
    df = df[['code', 'industry']].dropna()
    df = df[df['industry'] != '']
    # 合并到门类级：取行业代码首字母
    df['industry'] = df['industry'].str.extract(r'^([A-Z])')
    df = df.dropna(subset=['industry'])
    return df
    return df


def load_industry(force_refresh=False):
    """
    加载行业分类，缓存到 data/industry.pkl。
    返回 DataFrame: columns = ['symbol', 'industry']
    """
    if os.path.exists(INDUSTRY_CACHE) and not force_refresh:
        print(f"从缓存加载行业分类: {INDUSTRY_CACHE}")
        return pd.read_pickle(INDUSTRY_CACHE)

    print("拉取行业分类...")
    df = fetch_industry()
    df = df.rename(columns={'code': 'symbol'})
    df = df.drop_duplicates(subset='symbol', keep='last').reset_index(drop=True)

    pd.to_pickle(df, INDUSTRY_CACHE)
    print(f"行业分类已缓存: {INDUSTRY_CACHE}, 共 {len(df)} 只")
    return df


def neutralize_by_industry(factor_df, industry_df, factor_cols):
    """
    对每个因子做行业中性化：因子值 - 行业均值。
    返回新的 DataFrame，只改 factor_cols 列。
    """
    df = factor_df.copy()
    df = df.merge(industry_df, on='symbol', how='left')
    df['industry'] = df['industry'].fillna('UNKNOWN')

    for col in factor_cols:
        if col not in df.columns:
            continue
        # 行业均值
        ind_mean = df.groupby('industry')[col].transform('mean')
        # 减去行业均值
        df[col] = df[col] - ind_mean

    df = df.drop(columns=['industry'])
    return df


if __name__ == '__main__':
    ind = load_industry(force_refresh=True)
    print(ind.head(10))
    print(f"行业数: {ind['industry'].nunique()}")