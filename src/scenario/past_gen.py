
import datetime
import numpy as np
import pandas as pd
from .scenario_gen import ScenarioGen


class PastGen(ScenarioGen):
    """
    Implements azure blob file storage
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
 
    def gen_scenarios(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Return the data with the given starting_index and ending_index
        """
        if self.use_log_returns:
            returns = np.log(returns + 1)
        return returns
    
    # def get_summary(self,data:pd.DataFrame):

    #     """
    #     Return the summary stastics of given data starting_index and ending_index
    #     """
    #     series_mean = data.mean(axis=0)
    #     series_var = data.var(axis=0)
    #     series_std = data.std(axis=0)
    #     df_summary = pd.DataFrame([series_mean, series_var,series_std],index = ["mean","variance","std"])

    #     return df_summary
    
    # def get_covariance(self,data:pd.DataFrame):
    #     "Return Covariance matrix of the given data"
    #     return data.cov()
