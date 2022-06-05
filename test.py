import pandas as pd
from factorevaluation.multifactor import *
import time

# df = pd.read_pickle("data/all_stock_data_W.pkl")
# # df = df[df['交易日期'] >= pd.to_datetime("20160101")]

# df = df[['交易日期', '股票代码', '成交额std_5', '下周期每天涨跌幅']]
# df['next_ret'] = df['下周期每天涨跌幅'].apply(lambda x: np.prod(np.array(x) + 1) - 1)
# del df['下周期每天涨跌幅']
# df.dropna(inplace=True)
# s_time = time.time()
# factor = evaluation(df, '成交额std_5')
# run_full_func(factor)
# print(f"合计用时 - {time.time()-s_time} s")

df = pd.read_pickle("data/all_stock_data_W.pkl")
# df = df[df['交易日期'] >= pd.to_datetime("20160101")]
df['next_ret'] = df['下周期每天涨跌幅'].apply(lambda x: np.prod(np.array(x) + 1) - 1)
del df['下周期每天涨跌幅']
df.dropna(inplace=True)
factors = ['总市值', '换手率mean_20', '量价相关系数_5', '流通市值', '成交额std_5', '成交额std_20']
df = df[['交易日期', '股票代码', 'next_ret'] + factors]
s = time.time()
factor = multifactor(df, factor_list=factors)
factor.equal_weight_factor()
print(f"合计用时 - {time.time()-s} s")