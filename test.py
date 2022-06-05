import pandas as pd
from factorevaluation.one_factor import *
import multiprocessing, time

df = pd.read_pickle("data/all_stock_data_W.pkl")
# df = df[df['交易日期'] >= pd.to_datetime("20160101")]

df = df[['交易日期', '股票代码', '换手率mean_20', '下周期每天涨跌幅']]
df['next_ret'] = df['下周期每天涨跌幅'].apply(lambda x: np.prod(np.array(x) + 1) - 1)
del df['下周期每天涨跌幅']
df.dropna(inplace=True)
s_time = time.time()
factor = evaluation(df, '换手率mean_20')
run_full_func(factor)
print(f"合计用时 - {time.time()-s_time} s")