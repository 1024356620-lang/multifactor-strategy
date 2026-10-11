# valuation.py
import pandas as pd

def calc_valuation_factors(klines, fin_df, end_date):
    """
    计算 BP（账面市值比）和 EP（盈利收益率）。
    BP = 净资产 / 市值 ≈ (epsTTM / PB) 的倒数逻辑，这里用 MBRevenue 代理
    EP = 净利润TTM / 市值 = epsTTM / close
    """
    end_date = pd.to_datetime(end_date)
    fin_latest = fin_df[fin_df['pubDate'] <= end_date].sort_values('pubDate').groupby('code').tail(1)
    
    rows = []
    for _, row in fin_latest.iterrows():
        code = row['code']
        if code not in klines:
            continue
        df = klines[code]
        sub = df[df['date'] <= end_date]
        if len(sub) == 0:
            continue
        close = sub['close'].iloc[-1]
        
        # EP = epsTTM / close（越大越好，低估值）
        eps = row.get('epsTTM', None)
        ep = eps / close if eps and close > 0 else None
        
        # ROE
        roe = row.get('roeAvg', None)
        
        # 净利率
        npmargin = row.get('npMargin', None)
        
        rows.append({
            'symbol': code,
            'EP': ep,
        })
    return pd.DataFrame(rows)