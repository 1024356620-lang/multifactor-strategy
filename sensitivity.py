# sensitivity.py
import subprocess
import re
import pandas as pd

results = []
for w_ma in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    w_rev = 1.0 - w_ma
    # 改 config.py
    with open('config.py', 'r', encoding='utf-8') as f:
        content = f.read()
    content = re.sub(
        r"'reversal': [\d.]+,\s*\n\s*'volatility': [\d.]+,\s*\n\s*'avg_turn': [\d.]+,\s*\n\s*'ma_ratio': [\d.]+,",
        f"'reversal': {w_rev},\n    'volatility': 0.0,\n    'avg_turn': 0.0,\n    'ma_ratio': {w_ma},",
        content
    )
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(content)

    # 跑
    subprocess.run(['python', 'main.py'], capture_output=True)
    with open('results/long_short.csv') as f:
        pass
    # 从日志抓夏普
    # 简化：直接读 nav_net.csv 算
    nav = pd.read_csv('results/nav_net.csv')['nav_net']
    ls = pd.read_csv('results/long_short.csv')['ls_net']
    n = len(ls)
    ann_ret = nav.iloc[-1] ** (4/n) - 1
    ann_vol = ls.std(ddof=1) * 2
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    t = sharpe * (n/4) ** 0.5
    results.append({'w_ma': w_ma, 'w_rev': w_rev, 'sharpe': sharpe, 't': t})

df = pd.DataFrame(results)
print(df.to_string(index=False))
df.to_csv('results/sensitivity.csv', index=False)