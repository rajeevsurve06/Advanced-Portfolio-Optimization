
import pandas as pd
import numpy as np
from scipy import optimize
from abc import ABC, abstractmethod


## NSGA
import jax.numpy as jnp

from unittest.mock import patch
import pymoo.gradient.toolbox as anp

with patch("pymoo.gradient.toolbox", new=jnp):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.core.problem import Problem, ElementwiseProblem
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    # from pymoo.operators.selection.tournament import TournamentSelection
    # from pymoo.operators.selection.rnd import RandomSelection
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    # from pymoo.operators.mutation.pm import PolynomialMutation
    # from pymoo.operators.crossover.pntx import TwoPointCrossover
    # from pymoo.operators.mutation.bitflip import BitflipMutation
    # from pymoo.operators.sampling.rnd import BinaryRandomSampling
    # from pymoo.visualization.scatter import Scatter

from src.datasource.yahoodata import YahooDataSource
from src.scenario.scenario_gen import ScenarioGen


class Strategy(ABC):
    """
    Contains abstract methods for scenario generation
    """

    @abstractmethod
    def get_optimal_allocations(self,*args,**kwargs):
        """
        Get the Optimal weights 
        """

    @abstractmethod
    def run_strategy(self,*args,**kwargs):
        """
        Run strategy between this ind
        """

class ConstrainedBasedStrategy(Strategy):
    
    def calculate_VaR_CVaR(self,returns_window:pd.DataFrame,weights:np.array, alpha:float=0.99):
        """
        Calculate the Value at Risk
        """
        portfolio_returns = np.dot(returns_window,weights)
        VaR = np.percentile(portfolio_returns,100*(1-alpha))
        CVaR = np.mean(portfolio_returns[portfolio_returns<VaR])
        return -VaR, -CVaR
        

    def run_strategy(self, data_source:YahooDataSource, test_steps: int = 12, rebalancing_frequency_step: int = 1, start_date: str = None, end_date: str = None, data_frequency: str = '1MS', scenario_generator: ScenarioGen = None):

        weights_dict = {}
        VaR_dict = {}
        CVaR_dict = {}

        start_date = start_date if start_date else data_source.start_date
        end_date = end_date if end_date else data_source.end_date
        
        date_range = pd.date_range(start_date, end_date, freq=data_frequency)

        for index, date in enumerate(date_range[test_steps::rebalancing_frequency_step]):

            test_start_date = date_range[index*rebalancing_frequency_step]
            test_end_date = date

            price_data = data_source.get_data_by_frequency(start_date = test_start_date, end_date = test_end_date, frequency = data_frequency)
            
            rtn_data = price_data.pct_change()[1:]
            
            # generate scenarios using vine copula
            if scenario_generator:
                scenarios = scenario_generator.gen_scenarios(rtn_data)

            wealth_allocations = self.get_optimal_allocations(scenarios.T if scenario_generator else rtn_data.T,1)
            
            weights_dict[date] = dict(zip(price_data.columns,wealth_allocations))
            VaR_dict[date],CVaR_dict[date] = self.calculate_VaR_CVaR(rtn_data,wealth_allocations)

        weights_dict = pd.DataFrame(weights_dict).T
        weights_dict.index = pd.to_datetime(weights_dict.index)
        
        VaR_dict = pd.DataFrame(VaR_dict,index=["VaR"]).T
        CVaR_dict = pd.DataFrame(CVaR_dict,index=["CVaR"]).T
        
        return weights_dict, VaR_dict, CVaR_dict


class CvarMretOpt(ConstrainedBasedStrategy):


    def __init__(self,ratio=0.5,risk_level=0.1):


        self.ratio = ratio
        self.risk_level = risk_level
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.results = None
        

    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):
        self.array = returns_data.iloc[:, 1:].to_numpy()
        self.num_assets = len(self.array[:,0])
        self.num_senarios = len(self.array[0,:])
        self.array_transpose = np.transpose(self.array)
        self.investment_amount = investment_amount 
        self.results = self.optimize(self.ratio,self.risk_level)
        return self.results.x[self.num_senarios:self.num_senarios+self.num_assets]
    
    def get_cvar_value(self):
        return self.results.x[-1]

    def optimize(self,ratio,risk_level):

        """Solve the problem of minimizing the function 
                -(1-c) E[Z(x)] + c AVaR[Z(x)]
        """

        mean = np.resize(self.array.mean(axis=1),(self.num_assets,1))

        lhs_ineq = np.zeros((self.num_senarios,self.num_senarios+self.num_assets+1))

        for i in range(self.num_senarios):
            lhs_ineq[i,i] = -1  # vk
            lhs_ineq[i,self.num_senarios:self.num_senarios+self.num_assets] = -1*(self.array[:,i]) # Rk  
            lhs_ineq[i,-1] = 1    # n

        rhs_ineq = np.zeros((1,self.num_senarios))

        lhs_eq = np.zeros((1,self.num_senarios+self.num_assets+1))
        lhs_eq[0,self.num_senarios:self.num_senarios+self.num_assets] = 1
        rhs_eq = [self.investment_amount]

        # bnd = []
        # for i in range(self.num_senarios):
        #     bnd.append((0,float('inf')))

        # for i in range(self.num_assets+1):
        #     bnd.append((float("-inf"),float('inf')))

        # bnd[-1] = (float('-inf'),float('inf'))

        bnd = []
        for i in range(self.num_senarios):
            bnd.append((0,float('inf')))

        for i in range(self.num_assets+1):
            bnd.append((0,float('inf')))
            
        bnd[-1] = (float('-inf'),float('inf'))


        obj = np.ones((1,self.num_senarios+self.num_assets+1))*(1/risk_level)*(1/self.num_senarios)*(ratio)
        obj[0,-1] = -1*(ratio)
        obj[0,self.num_senarios:self.num_senarios+self.num_assets] = -1*(1-ratio)*np.array(np.transpose(mean))
        opt = optimize.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,A_eq=lhs_eq,b_eq=rhs_eq, bounds=bnd,method="highs")

        return opt

        

class MeanSemidevOpt(ConstrainedBasedStrategy):

    def __init__(self,ratio):
       
        self.ratio = ratio 
        self.results = None
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.results = None
        
    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):

        self.array = returns_data.iloc[:, 1:].to_numpy()
        self.num_assets = len(self.array[:,0])
        self.num_senarios = len(self.array[0,:])
        self.array_transpose = np.transpose(self.array)
        self.investment_amount = investment_amount
        self.results = self.optimize(self.ratio)
        return self.results.x[self.num_senarios:self.num_senarios+self.num_assets]

    def optimize(self,ratio):

        """
            ρ =  Hybrid Risk Measure
            ρ = [Z(x)] = -E[Z(x)] + c σ[Z(x)]

            where σ[Z] = E{ max(0,E[Z] – Z)} is the lower semideviation of the first order.
        """

        mean = np.resize(self.array.mean(axis=1),(self.num_assets,1))
        lhs_ineq = np.zeros((self.num_senarios,self.num_senarios+self.num_assets))

        for i in range(self.num_senarios):
            lhs_ineq[i,i] = -1  # vk
            lhs_ineq[i,self.num_senarios:self.num_senarios+self.num_assets] = -1*(self.array[:,i]) + np.transpose(mean)  # Rk

        rhs_ineq = np.zeros((1,self.num_senarios))

        lhs_eq = np.zeros((1,self.num_senarios+self.num_assets))
        lhs_eq[0,self.num_senarios:self.num_senarios+self.num_assets] = 1
        rhs_eq = [self.investment_amount]

        bnd = []
        for i in range(self.num_senarios+self.num_assets):
            bnd.append((0,float('inf')))

        obj = np.ones((1,self.num_senarios+self.num_assets))*(1/self.num_senarios)*ratio
        obj[0,self.num_senarios:self.num_senarios+self.num_assets] = -1*np.array(np.transpose(mean))

        opt = optimize.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,A_eq=lhs_eq,b_eq=rhs_eq, bounds=bnd,method="highs")
        return opt
    

class EqualyWeighted(ConstrainedBasedStrategy):

    def __init__(self):
    
        self.results = None
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.results = None

    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):

        self.array = returns_data.iloc[:, 1:].to_numpy()
        self.num_assets = len(self.array[:,0])
        return (np.ones(self.num_assets)/self.num_assets)*investment_amount
    


class MeanSemiEquallyWeighted(ConstrainedBasedStrategy):

    def __init__(self,ratio):
       
        self.ratio = ratio 
        self.results = None
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.results = None
        
    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):

        self.array = returns_data.iloc[:, 1:].to_numpy()
        self.num_assets = len(self.array[:,0])
        self.num_senarios = len(self.array[0,:])
        self.array_transpose = np.transpose(self.array)
        self.investment_amount = 1
        self.results = self.optimize(self.ratio)
        allocations = np.array(self.results.x[self.num_senarios:self.num_senarios+self.num_assets])
        return (allocations + (np.ones(self.num_assets)/self.num_assets))/(np.sum(allocations) + (np.sum((np.ones(self.num_assets)/self.num_assets))))

    def optimize(self,ratio):

        """
            ρ =  Hybrid Risk Measure
            ρ = [Z(x)] = -E[Z(x)] + c σ[Z(x)]

            where σ[Z] = E{ max(0,E[Z] – Z)} is the lower semideviation of the first order.
        """

        mean = np.resize(self.array.mean(axis=1),(self.num_assets,1))
        lhs_ineq = np.zeros((self.num_senarios,self.num_senarios+self.num_assets))

        for i in range(self.num_senarios):
            lhs_ineq[i,i] = -1  # vk
            lhs_ineq[i,self.num_senarios:self.num_senarios+self.num_assets] = -1*(self.array[:,i]) + np.transpose(mean)  # Rk

        rhs_ineq = np.zeros((1,self.num_senarios))

        lhs_eq = np.zeros((1,self.num_senarios+self.num_assets))
        lhs_eq[0,self.num_senarios:self.num_senarios+self.num_assets] = 1
        rhs_eq = [self.investment_amount]

        bnd = []
        for i in range(self.num_senarios+self.num_assets):
            bnd.append((0,float('inf')))

        obj = np.ones((1,self.num_senarios+self.num_assets))*(1/self.num_senarios)*ratio
        obj[0,self.num_senarios:self.num_senarios+self.num_assets] = -1*np.array(np.transpose(mean))

        opt = optimize.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,A_eq=lhs_eq,b_eq=rhs_eq, bounds=bnd,method="highs")
        return opt

    


class StochasticDominant(ConstrainedBasedStrategy):

    def __init__(self,ratio):
       
        self.ratio = ratio
        self.results = None
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.results = None
        
    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):

        self.array = returns_data.iloc[:, 1:].to_numpy()
        self.num_assets = len(self.array[:,0])
        self.num_senarios = len(self.array[0,:])
        self.array_transpose = np.transpose(self.array)
        self.investment_amount = investment_amount
        try:
            self.results = self.optimize()
            allocations = np.array(np.array(self.results.x)[:self.num_assets])
            return (allocations + (np.ones(self.num_assets)/self.num_assets))/(np.sum(allocations) + (np.sum((np.ones(self.num_assets)/self.num_assets))))
        except:
            self.array = returns_data.iloc[:, 1:].to_numpy()
            self.num_assets = len(self.array[:,0])
            self.num_senarios = len(self.array[0,:])
            self.array_transpose = np.transpose(self.array)
            self.investment_amount = 1
            self.results = self.optimize(self.ratio)
            allocations = np.array(self.results.x[self.num_senarios:self.num_senarios+self.num_assets])
            return (allocations + (np.ones(self.num_assets)/self.num_assets))/(np.sum(allocations) + (np.sum((np.ones(self.num_assets)/self.num_assets))))


    def optimize(self):

        # """
        #     ρ =  Hybrid Risk Measure
        #     ρ = [Z(x)] = -E[Z(x)] + c σ[Z(x)]

        #     where σ[Z] = E{ max(0,E[Z] – Z)} is the lower semideviation of the first order.
        # """

        equally_weighted = (np.ones((1,self.num_assets))/self.num_assets)[0,:]
        values = equally_weighted@self.array

        eq_dict = sorted(dict(zip(values,self.array.T)).items())

        lhs_ineq = np.zeros((self.num_senarios , self.num_senarios+self.num_assets))
        rhs_ineq = np.zeros((1,self.num_senarios))

        for ind,dict_item in enumerate(eq_dict):
            loss,senario = dict_item
            lhs_ineq[ind, :self.num_assets] = -senario
            lhs_ineq[ind, self.num_assets+ind] = +1
            if loss<0:
                rhs_ineq[:,ind] = -loss
            else:
                rhs_ineq[:,ind] = -np.array((sorted(values[values>0])[0]))

        rhs_ineq = np.zeros((1,self.num_senarios))

        lhs_eq = np.zeros((1,self.num_senarios+self.num_assets))
        lhs_eq[0,:self.num_assets] = 1
        rhs_eq = [self.investment_amount]

        bnd = []
        for i in range(self.num_assets):
            bnd.append((float("-inf"),float('inf')))

        for i in range(self.num_senarios):
            bnd.append((float("0"),float('inf')))

        obj = np.zeros((1,self.num_senarios+self.num_assets))
        obj[0,self.num_assets:] = -1

        opt = optimize.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,A_eq=lhs_eq,b_eq=rhs_eq, bounds=bnd,method="highs")
        return opt


    def optimize_2(self,ratio):

        """
            ρ =  Hybrid Risk Measure
            ρ = [Z(x)] = -E[Z(x)] + c σ[Z(x)]

            where σ[Z] = E{ max(0,E[Z] – Z)} is the lower semideviation of the first order.
        """

        mean = np.resize(self.array.mean(axis=1),(self.num_assets,1))
        lhs_ineq = np.zeros((self.num_senarios,self.num_senarios+self.num_assets))

        for i in range(self.num_senarios):
            lhs_ineq[i,i] = -1  # vk
            lhs_ineq[i,self.num_senarios:self.num_senarios+self.num_assets] = -1*(self.array[:,i]) + np.transpose(mean)  # Rk

        rhs_ineq = np.zeros((1,self.num_senarios))

        lhs_eq = np.zeros((1,self.num_senarios+self.num_assets))
        lhs_eq[0,self.num_senarios:self.num_senarios+self.num_assets] = 1
        rhs_eq = [self.investment_amount]

        bnd = []
        for i in range(self.num_senarios+self.num_assets):
            bnd.append((0,float('inf')))

        obj = np.ones((1,self.num_senarios+self.num_assets))*(1/self.num_senarios)*ratio
        obj[0,self.num_senarios:self.num_senarios+self.num_assets] = -1*np.array(np.transpose(mean))

        opt = optimize.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,A_eq=lhs_eq,b_eq=rhs_eq, bounds=bnd,method="highs")
        return opt

class MeanVariance(ConstrainedBasedStrategy):

    
    def __init__(self,target_return=None,allow_shorting=False):

        self.results = None
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.target_return =  target_return
        self.allow_shorting = allow_shorting
        
    def calculate_mean(self) -> pd.Series:
        mean_returns = self.returns.mean()
        return mean_returns

    def calculate_covariance_matrix(self) -> pd.DataFrame:
        covariance_matrix = self.returns.cov()
        return covariance_matrix

    def portfolio_variance(self, weights):
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))
    
    def min_variance(self, allow_shorting=False):
        num_assets = len(self.mean_returns)
        initial_weights = np.ones(num_assets) / num_assets
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        if allow_shorting:
            bounds = tuple((float('-inf'), float('inf')) for _ in range(num_assets))  
        else:
            bounds = tuple((0, float('inf')) for _ in range(num_assets))  
        
        result = optimize.minimize(self.portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result
    
    def min_variance_allocation(self):
        
        result = self.min_variance()
        if result.success:
            return dict(zip(self.returns.columns, result.x))
        else:
            raise ValueError("Optimization failed: " + result.message)

    def get_max_return(self) -> float:
        return self.mean_returns.max()

    def minimize_func(self, weights):
        return np.matmul(np.matmul(np.transpose(weights), self.cov_matrix), weights)
    
    def optimize(self, target_return=None, allow_shorting=False):
        
        num_assets = len(self.mean_returns)
        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
        if target_return is not None:
            constraints.append({'type': 'eq', 'fun': lambda weights: np.dot(weights, self.mean_returns) - target_return})
        
        if allow_shorting:
            bounds = tuple((float('-inf'), float('inf')) for _ in range(num_assets))
        else:
            bounds = tuple((0, float('inf')) for _ in range(num_assets))
        
        initial_weights = np.ones(num_assets) / num_assets
        result = optimize.minimize(self.minimize_func, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):

        self.returns =  returns_data.T
        self.mean_returns = self.calculate_mean().to_numpy()
        self.cov_matrix = self.calculate_covariance_matrix().to_numpy()

        result = self.optimize(target_return=self.target_return,allow_shorting=self.allow_shorting)
        if result.success:
            return result.x
        else:
            raise ValueError("Optimization failed: " + result.message)
        
class NSGA(ConstrainedBasedStrategy):
    
    class NSGAProblem(Problem):
        def __init__(self, ret, **kwargs):
            super().__init__(n_var=ret.shape[0], n_obj=2, n_eq_constr=1, xl=-1, xu=1, **kwargs)
            self.ret = ret

        def _evaluate(self, x, out, *args, **kwargs):
            
            # Objective functions
            obj_1 = -np.dot(np.mean(self.ret, axis=1), x.T)  # Shape (pop_size,)
            obj_2 = np.einsum('ij,jk,ik->i', x, np.cov(self.ret, ddof=0), x)  # Shape (pop_size,)
            
            # Constraints
            constr_1 = np.sum(x, axis=1).reshape(-1, 1) - 1  # Shape (pop_size,)

            # Assigning the values to the output
            out["F"] = np.column_stack([-obj_1, obj_2])
            out["H"] = constr_1

    def __init__(self, pop_size = 40, n_offsprings = 20, sampling = FloatRandomSampling(), crossover = SBX(prob=0.9, eta=15), mutation = PM(eta=20), eliminate_duplicates = True):

        self.algorithm = NSGA2(pop_size=pop_size,
                                 n_offsprings=n_offsprings,
                                 sampling=sampling,
                                 crossover=crossover,
                                 mutation=mutation,
                                 eliminate_duplicates=eliminate_duplicates)
        
        self.stop_criteria = get_termination("n_gen", 100)

    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):

        self.ret = returns_data.iloc[:, 1:].to_numpy()
        problem = self.NSGAProblem(self.ret)
        results = minimize(problem = problem, algorithm = self.algorithm, termination = self.stop_criteria, seed=1, save_history=True, verbose=False)
        return results.X.flatten()
