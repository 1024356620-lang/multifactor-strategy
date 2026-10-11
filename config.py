# config.py
import os

# 数据缓存目录
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 回测参数
START_DATE = '2020-01-01'
END_DATE = '2025-10-01'
STOCK_POOL_SIZE = 300          # 先取前50只测试，可改为300
REBALANCE_FREQ = 'M'          # 月度调仓
MIN_STOCKS_PER_PERIOD = 20    # 每期最少有效股票数
FACTOR_MIN_HISTORY = 21       # 因子计算最少历史天数
# config.py 追加
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-06-30'
VALID_START = '2023-07-01'
VALID_END = '2025-10-01'

# 因子列表（与 factor_calc.py 保持一致）
FACTOR_COLS = ['momentum', 'volatility', 'avg_turn', 'ma_ratio', 'EP']

# 因子方向：1 表示越大越好，-1 表示越小越好
FACTOR_DIRECTION = {
    'momentum': -1,      
    'volatility': -1,    
    'avg_turn': -1,      
    'ma_ratio': -1,      
    'EP': 1,             
}

# 因子合成权重（可调整）
FACTOR_WEIGHTS = {
    'momentum': 0.24,
    'volatility': 0.28,
    'avg_turn': 0.24,
    'ma_ratio': 0.39,
    'EP': 0.15,
}

# 交易成本（单边）
TRANSACTION_COST = 0.0025