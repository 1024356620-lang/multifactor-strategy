# config.py
import os

# 数据缓存目录
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 回测参数
START_DATE = '2020-01-01'
END_DATE = '2026-09-30'         # 拉到 2026Q3 结束
STOCK_POOL_SIZE = 300
REBALANCE_FREQ = 'Q'            # 季度调仓
PERIODS_PER_YEAR = 4            # 年化因子：季度=4，月度=12，日频=252
MIN_STOCKS_PER_PERIOD = 20
FACTOR_MIN_HISTORY = 21

TRAIN_START = '2020-01-01'
TRAIN_END = '2023-06-30'
VALID_START = '2023-07-01'
VALID_END = '2026-09-30'        # 顺延到 2026Q3 结束，确保验证期包含新数据

# 因子列表（与 factor_calc.py 保持一致）
FACTOR_COLS = ['reversal', 'volatility', 'avg_turn', 'ma_ratio', 'EP',]

# 因子方向：1 表示越大越好，-1 表示越小越好
FACTOR_DIRECTION = {
    'reversal': 1,      
    'volatility': 1,    
    'avg_turn': 1,      
    'ma_ratio': -1,      
    'EP': 1,        
}

# 因子合成权重
FACTOR_WEIGHTS = {
    'reversal': 0.4,
    'volatility': 0.0,
    'avg_turn': 0.0,
    'ma_ratio': 0.6,
    'EP': 0.0,
}

# 交易成本（单边）
TRANSACTION_COST = 0.0025

# 账本与异常
LEDGER_DIR = os.path.join(os.path.dirname(__file__), 'output', 'ledger')
ATTRIBUTION_DIR = os.path.join(os.path.dirname(__file__), 'output', 'attribution')
DAILY_NAV_DIR = os.path.join(os.path.dirname(__file__), 'output', 'daily_nav')
EXPERIMENT_DIR = os.path.join(os.path.dirname(__file__), 'experiments')
for _d in [LEDGER_DIR, ATTRIBUTION_DIR, DAILY_NAV_DIR, EXPERIMENT_DIR]:
    os.makedirs(_d, exist_ok=True)

# 成本假设（按成交额计费，用于账本对账）
COMMISSION_RATE = 0.0025      # 单边总费率（含佣金+滑点）
STAMP_TAX_RATE = 0.0          # 印花税并入 COMMISSION_RATE，避免重复
SLIPPAGE_BPS = 0
BORROW_RATE_ANNUAL = 0.08     # 融券年化费率（用于归因，不直接扣减）

# 异常状态枚举
STATUS_FILLED = "filled"
STATUS_UNFILLED_SUSPENDED = "unfilled_suspended"
STATUS_HELD_SUSPENDED = "held_suspended"
STATUS_DELISTED = "delisted"
STATUS_DATA_ERROR = "data_error"
STATUS_NO_VALUATION = "no_valuation"

# 基线实验开关（默认关闭，避免误跑）
RUN_BASELINE = False