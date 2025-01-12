import pandas as pd
import numpy as np
from copulas.multivariate import GaussianMultivariate, VineCopula, Multivariate
from typing import Literal, List

from .scenario_gen import ScenarioGen

class CopulaGen(ScenarioGen):
    """
    Implements azure blob file storage
    """

    def __init__(self, copula_type: Literal["gaussian", "t", "clayton", "gumbel", "frank", "vine-center", "vine-direct"] = "gaussian", *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.copula_type = copula_type
        
        if copula_type == "gaussian":
            self.copula = GaussianMultivariate()
        elif copula_type == "vine-center":
            self.copula = VineCopula("center")
        elif copula_type == "vine-direct":
            self.copula = VineCopula("direct")
        else:
            raise ValueError("Invalid copula type")
            
 
    def gen_scenarios(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Return the data with the given starting_index and ending_index
        """
        if self.use_log_returns:
            returns = np.log(returns + 1)
        self.copula.fit(returns)
        return self.copula.sample(self.sample_size if self.sample_size > 0 else returns.shape[0])