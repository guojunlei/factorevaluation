"""
用于但因子评价
author:guojunlei
date:2022.06.04
"""

import re
from statistics import mode
import pandas as pd
import numpy as np
import statsmodels.api as sm


class evaluation:

    def __init__(self, df: pd.DataFrame, factor: str) -> None:
        """
        data:pd.DataFrame
        factor: factor name
        """
        self.data = df
        self.factors = factor
        self.ICstat = pd.DataFrame()
        self.REGstat = pd.DataFrame()

    @classmethod
    def _func_icir(cls, x, name):
        return x[name].corr(x['next_ret'])

    @classmethod
    def _regression_rlm(cls, X, name):
        X[name] = (X[name] - X[name].mean()) / X[name].std()
        x = X[name].values
        y = X['next_ret'].values
        x = sm.add_constant(x)
        model = sm.RLM(y, x, M=sm.robust.norms.HuberT()).fit()
        return model.params[-1], model.tvalues[-1]

    def calculate_ICIR(self):
        df = self.data
        ic_data = df.groupby('交易日期').apply(self._func_icir, self.factors)
        mean_ic = np.mean(ic_data)
        if mean_ic > 0:
            ic_pr = ic_data[ic_data > 0].shape[0] / ic_data.shape[0]
        else:
            ic_pr = ic_data[ic_data < 0].shape[0] / ic_data.shape[0]
        ir = abs(mean_ic) / ic_data.std()

        self.ICstat.loc[self.factors, '因子IC均值'] = mean_ic
        self.ICstat.loc[self.factors, 'IC_概率(均值>0或小于0)'] = ic_pr
        if abs(mean_ic) > 0.02:
            self.ICstat.loc[self.factors,
                            '因子IC评价(大于0.02(可用)大于0.04(较好))'] = '可用'
        elif abs(mean_ic) > 0.04:
            self.ICstat.loc[self.factors,
                            '因子IC评价(大于0.02(可用)大于0.04(较好))'] = '较好'
        else:
            self.ICstat.loc[self.factors,
                            '因子IC评价(大于0.02(可用)大于0.04(较好))'] = '不好'
        self.ICstat.loc[self.factors, '因子IC绝对值>0.02的概率'] = ic_data[
            abs(ic_data) > 0.02].shape[0] / ic_data.shape[0]
        self.ICstat.loc[self.factors, '因子IC绝对值>0.04的概率'] = ic_data[
            abs(ic_data) > 0.04].shape[0] / ic_data.shape[0]
        self.ICstat.loc[self.factors, '因子IR'] = ir
        self.ICstat = self.ICstat.T

    def regression_method(self):
        df = self.data
        factor_k = df.groupby('交易日期').apply(self._regression_rlm, self.factors)
        factor_k = pd.DataFrame(factor_k)
        factor_k['k'] = factor_k[0].apply(lambda x: x[0])
        factor_k['t'] = factor_k[0].apply(lambda x: x[1])
        del factor_k[0]
        ret_mean = factor_k['k'].mean()
        ret_t = np.abs(ret_mean) / factor_k['k'].std()
        factor_t_abs = factor_k['t'].abs().mean()
        self.REGstat.loc[self.factors, '因子平均收益'] = ret_mean
        self.REGstat.loc[self.factors, '因子收益t值'] = ret_t
        self.REGstat.loc[self.factors, 't值绝对值'] = factor_t_abs
        if ret_mean > 0:
            self.REGstat.loc[self.factors, '因子均值大于或小于0的概率'] = factor_k[
                factor_k['k'] > 0].shape[0] / factor_k.shape[0]
            self.REGstat.loc[self.factors, '因子均值大于或小于0t值大于2的概率'] = factor_k[
                factor_k['t'] > 2].shape[0] / factor_k.shape[0]
        else:
            self.REGstat.loc[self.factors, '因子均值大于或小于0的概率'] = factor_k[
                factor_k['k'] < 0].shape[0] / factor_k.shape[0]
            self.REGstat.loc[self.factors, '因子均值大于或小于0t值大于2的概率'] = factor_k[
                factor_k['t'] < -2].shape[0] / factor_k.shape[0]
        self.REGstat=self.REGstat.T

    def grouping(self):
        ...


if __name__ == "__main__":
    df = pd.read_pickle("../data/all_stock_data_W.pkl")
    df = df[df['交易日期'] <= pd.to_datetime("20160101")]
    df = df[['交易日期', '股票代码', '总市值', '下周期每天涨跌幅']]
    df['next_ret'] = df['下周期每天涨跌幅'].apply(
        lambda x: np.prod(np.array(x) + 1) - 1)
    del df['下周期每天涨跌幅']
    df.dropna(inplace=True)
    factor = evaluation(df, '总市值')
    factor.regression_method()
