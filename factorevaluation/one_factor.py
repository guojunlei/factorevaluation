"""
用于但因子评价
author:guojunlei
date:2022.06.04
"""

import pandas as pd

class evaluation:

    def __init__(self,df:pd.DataFrame,factor:str) -> None:
        """
        data:pd.DataFrame
        factor: factor name
        """
        self.data=df
        self.factors=factor

    def calculate_ICIR(self):
        
