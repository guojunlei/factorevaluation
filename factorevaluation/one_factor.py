"""
用于但因子评价
author:guojunlei
date:2022.06.04
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import multiprocessing


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
        self.GROUPstat = pd.DataFrame()
        self.RETstat = pd.DataFrame()

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

    @classmethod
    def _func_group(cls, x, name, num):
        x['group'] = pd.qcut(x[name].rank(method='first'), q=num, labels=False)
        ret_list = []
        for i in range(num):
            ret = x.loc[x['group'] == i, 'next_ret'].mean()
            ret_list.append(ret)
        return ret_list

    @classmethod
    def _cal_group(cls, x):
        df = pd.DataFrame(x)
        num = len(df.iloc[0, 0])
        ret_df = pd.DataFrame(index=df.index)
        for i in range(num):
            ret_df[f"group_{i+1}"] = df[0].apply(lambda x: x[i])
        final_ret = (ret_df + 1).cumprod()
        final_ret = final_ret.iloc[-1].tolist()
        if final_ret[0] < final_ret[-1]:
            label = '正'
            ret_df[
                'group_longshort'] = ret_df[f'group_{num}'] - ret_df["group_1"]
        else:
            label = '反'
            ret_df[
                'group_longshort'] = ret_df["group_1"] - ret_df[f'group_{num}']

        return ret_df, label

    def calculate_ICIR(self):
        df = self.data.copy()
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
        self.ICstat = self.ICstat

    def regression_method(self):
        df = self.data.copy()
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
        self.REGstat = self.REGstat

    def grouping(self, ngroup: int):
        df = self.data.copy()
        group_ret = df.groupby('交易日期').apply(self._func_group, self.factors,
                                             ngroup)
        group_ret, labal = self._cal_group(group_ret)
        self.GROUPstat.loc[self.factors, '因子方向'] = labal
        self.GROUPstat.loc[self.factors,
                           '多空收益'] = (group_ret['group_longshort'] +
                                      1).prod() - 1
        self.GROUPstat.loc[self.factors, '多空最大回撤'] = (
            (group_ret['group_longshort'] + 1).cumprod() /
            (group_ret['group_longshort'] + 1).cumprod().expanding().max() -
            1).min()
        group_ret.index = pd.to_datetime(group_ret.index)
        self.GROUPstat.loc[self.factors, '多空夏普'] = (
            group_ret['group_longshort'] + 1).cumprod().diff().mean() / (
                group_ret['group_longshort'] +
                1).cumprod().diff().std() * np.sqrt(
                    365 / (group_ret.index.tolist()[1] -
                           group_ret.index.tolist()[0]).days)

        self.RETstat = group_ret

    def draw_picture(self, if_save=True):
        df = self.RETstat.copy()
        fig1 = plt.figure(figsize=(18, 9))
        ax1 = plt.subplot(2, 1, 1)
        ax1.set_title("group_equity")
        for col in df.columns:
            plt.plot((df[col] + 1).cumprod(), label=col)
        plt.legend(loc='best')

        ax2 = plt.subplot(2, 1, 2)
        ax2.set_title("group_net")
        x = df.columns.tolist()
        y = np.array((df + 1).prod().tolist())
        plt.bar(x, y)
        plt.legend(loc='best')
        if if_save:
            if not os.path.exists(f"{os.getcwd()}/data/output/{self.factors}"):
                os.mkdir(f"{os.getcwd()}/data/output/{self.factors}")

            plt.savefig(
                f"{os.getcwd()}/data/output/{self.factors}/{self.factors}.jpg")
            res = pd.concat([self.ICstat, self.REGstat, self.GROUPstat],
                            axis=1)
            res.to_csv(
                f"{os.getcwd()}/data/output/{self.factors}/{self.factors}.csv",
                index=0)

        plt.show()


def run_full_func(factor, if_pro=True):
    if if_pro:
        process1 = multiprocessing.Process(target=factor.calculate_ICIR())
        process2 = multiprocessing.Process(target=factor.regression_method())
        process3 = multiprocessing.Process(target=factor.grouping(5))

        process1.start()
        process2.start()
        process3.start()

        process1.join()
        process1.join()
        process1.join()

        factor.draw_picture()
    else:
        factor.calculate_ICIR()
        factor.regression_method()
        factor.grouping(5)
        factor.draw_picture()


if __name__ == "__main__":
    ...
