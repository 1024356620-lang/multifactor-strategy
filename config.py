# config.py
import os

# 数据缓存目录
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 回测参数
START_DATE = '2020-01-01'
END_DATE = '2025-10-01'
STOCK_POOL_SIZE = 300
REBALANCE_FREQ = 'Q'          # 季度调仓
PERIODS_PER_YEAR = 4          # 年化因子：季度=4，月度=12，日频=252
MIN_STOCKS_PER_PERIOD = 20
FACTOR_MIN_HISTORY = 21
# config.py 追加
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-06-30'
VALID_START = '2023-07-01'
VALID_END = '2025-10-01'

# 因子列表（与 factor_calc.py 保持一致）
FACTOR_COLS = ['reversal', 'volatility', 'avg_turn', 'ma_ratio', 'EP']

# 因子方向：1 表示越大越好，-1 表示越小越好
FACTOR_DIRECTION = {
    'reversal': 1,      
    'volatility': 1,    
    'avg_turn': 1,      
    'ma_ratio': -1,      
    'EP': 1,             
}

# 因子合成权重（可调整）
FACTOR_WEIGHTS = {
    'reversal': 0.4,
    'volatility': 0.0,
    'avg_turn': 0.0,
    'ma_ratio': 0.6,
    'EP': 0.0,
}

# 交易成本（单边）
TRANSACTION_COST = 0.0025