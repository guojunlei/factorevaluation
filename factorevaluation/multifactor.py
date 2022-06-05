import pandas as pd
import numpy as np
import itertools, os, time
from factorevaluation.one_factor import *


class multifactor:

    def __init__(self, data: pd.DataFrame, factor_list: list) -> None:
        self.data = data
        self.factor_list = factor_list

    @classmethod
    def _calculate_corr(cls, x: pd.DataFrame, combin):
        x.sort_values(['交易日期'], inplace=True)
        com_dict = {}
        for com in combin:
            corr = x[com[0]].corr(x[com[1]])
            com_dict[com] = corr
        return com_dict

    @classmethod
    def _find_one_factor_corr(cls, name: tuple, corr: dict):
        res = {}
        for n in name:
            corr_list = []
            for k, v in corr.items():
                if (n in k) and (name != k):
                    corr_list.append(v)
            res[n] = np.mean(corr_list)
        if res[name[0]] > res[name[1]]:
            return name[0]
        else:
            return name[1]

    @classmethod
    def _delete_corr(cls, name, corr):
        new_corr = {}
        for k, v in corr.items():
            if name not in k:
                new_corr[k] = v
        return new_corr

    @classmethod
    def _factor_label(cls, name):
        if not os.path.exists(
                f"/home/tradingking/python/factorevaluation/data/output/{name}"
        ):
            print(f"{name}文件不存在")
            exit()
        df = pd.read_csv(
            f"/home/tradingking/python/factorevaluation/data/output/{name}/{name}.csv"
        )
        label = df['因子方向'].iat[0]
        return label

    @classmethod
    def _calculate_new_factor(cls, df: pd.DataFrame, label: dict):
        for k, v in label.items():
            if v == '正':
                df[f"{k}_z"] = (df[k] - df[k].mean()) / df[k].std()
            else:
                df[f"{k}_z"] = -((df[k] - df[k].mean()) / df[k].std())
        new_list = [name + "_z" for name in label.keys()]

        df[f"{'_'.join(label.keys())}"] = df[new_list].sum(axis=1)
        return df[['交易日期', '股票代码', 'next_ret'] + [f"{'_'.join(label.keys())}"]]

    # 1.等权
    def equal_weight_factor(self):
        df = self.data.copy()
        combin = list(itertools.combinations(self.factor_list, 2))
        corr_df = df.groupby('股票代码').apply(self._calculate_corr, combin)
        corr_dict = {}
        for com in combin:
            corr = corr_df.apply(lambda x: x[com])
            corr_dict[com] = corr.mean()
        del_list = []
        for k, v in corr_dict.items():
            for j in k:
                if j in del_list:
                    continue
            if v >= 0.5:
                del_name = self._find_one_factor_corr(k, corr_dict)
                corr_dict = self._delete_corr(del_name, corr_dict)
                del_list.append(del_name)
        new_factor = []
        for name in self.factor_list:
            if name not in del_list:
                new_factor.append(name)
        print(f"因子组合列表:{new_factor}")
        factor_label = {}
        for name in new_factor:
            factor_label[name] = self._factor_label(name)
        df = df[['交易日期', '股票代码', 'next_ret'] + new_factor]
        new_df = df.groupby('交易日期').apply(self._calculate_new_factor,
                                          factor_label)
        new_name = new_df.columns.tolist()[-1]
        one = evaluation(new_df, new_name)
        run_full_func(one)


if __name__ == "__main__":
    ...