# data_loader.py
import baostock as bs
import pandas as pd
import os
import time
from config import DATA_DIR, START_DATE, END_DATE, STOCK_POOL_SIZE

def get_zz500_stocks(n=300):
    """获取中证500成分股"""
    bs.login()
    rs = bs.query_zz500_stocks()  # baostock 有此接口
    stocks = []
    while (rs.error_code == '0') & rs.next():
        stocks.append(rs.get_row_data())
    bs.logout()
    df = pd.DataFrame(stocks, columns=['updateDate', 'code', 'code_name'])
    return df['code'].tolist()[:n]

def _fetch_one(code, start_date, end_date):
    """从 baostock 拉取单只股票后复权日线"""
    bs.login()
    fields = "date,open,high,low,close,volume,amount,turn,pctChg"
    rs = bs.query_history_k_data_plus(
        code, fields,
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="1"
    )
    rows = []
    while (rs.error_code == '0') & rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=fields.split(','))
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)

def load_all_data(force_refresh=False):
    """
    加载所有股票数据。若缓存存在且不强制刷新，直接读缓存。
    返回：dict {code: DataFrame}
    """
    cache_file = os.path.join(DATA_DIR, 'klines.pkl')
    
    if os.path.exists(cache_file) and not force_refresh:
        print(f"从缓存加载: {cache_file}")
        return pd.read_pickle(cache_file)
    
    symbols = get_zz500_stocks()
    print(f"股票池: {len(symbols)} 只")
    
    klines = {}
    for i, sym in enumerate(symbols):
        df = _fetch_one(sym, START_DATE, END_DATE)
        if df is not None and len(df) > 30:
            klines[sym] = df
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i+1}/{len(symbols)}")
        time.sleep(0.3)
    
    print(f"成功获取 {len(klines)} 只股票数据")
    pd.to_pickle(klines, cache_file)
    print(f"已缓存至: {cache_file}")
    return klines

if __name__ == '__main__':
    data = load_all_data(force_refresh=False)
    print(f"数据股票数: {len(data)}")
    sample = list(data.keys())[0]
    print(f"示例 {sample}:\n{data[sample].tail(3)}")