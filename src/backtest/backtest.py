import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from src.algorithms.strategy import Strategy
from src.datasource.yahoodata import YahooDataSource
from src.scenario.scenario_gen import ScenarioGen

class BackTest:

    def __init__(self, data_source: YahooDataSource, method:Strategy):
        self.data_source = data_source
        self.method = method

    # backtest the strategy and store the results (weights) in the backtest_results attribute
    def backtest(self, start_date: str, end_date: str, test_steps: int, rebalancing_frequency_step: int, data_frequency: str, scenario_generator: ScenarioGen = None):
        """ Backtest the strategy and store the results (portfolio value)

        Args:
            start_date (str): start date of the backtest period
            end_date (str): end date of the backtest period
            test_steps (int): number of steps to run the strategy
            rebalancing_frequency_step (int): frequency of rebalancing the portfolio
            data_frequency (str): frequency of the data
            scenario_generator (ScenarioGen): instance of the scenario generator
        """
        
        # run the strategy to get the weights for the backtest period for each rebalancing period
        allocated_weights, VaR_historical, CVaR_historical = self.method.run_strategy(self.data_source, test_steps, rebalancing_frequency_step, start_date, end_date, data_frequency, scenario_generator)
        
        # fetch the data for the backtest period and calculate the returns
        data = self.data_source.get_data_by_frequency(start_date, end_date, "1d")
        data.ffill(inplace=True)
        
        # calculate the stock holding quantity and portfolio value for each day in the backtest period
        stock_holding_quantity = pd.DataFrame(index=data.index, columns=data.columns)
        portfolio_value = pd.DataFrame(index=data.index, columns=["Portfolio Value"])
        VaR_value = pd.DataFrame(index=data.index, columns=["VaR"])
        CVaR_value = pd.DataFrame(index=data.index, columns=["CVaR"])
        for idx, date in enumerate(data.index):
            if idx == 0:
                portfolio_value.loc[date] = 1
                VaR_value.loc[date] = 0
                CVaR_value.loc[date] = 0
            elif stock_holding_quantity.iloc[idx - 1].sum() == 0:
                portfolio_value.loc[date] = portfolio_value.iloc[idx - 1]
            else:
                portfolio_value.loc[date] = (stock_holding_quantity.iloc[idx - 1] * data.loc[date]).sum()
                
            if date in allocated_weights.index:
                stock_holding_quantity.loc[date] = (allocated_weights.loc[date] / data.loc[date]) * portfolio_value.loc[date, "Portfolio Value"]
            else:
                stock_holding_quantity.loc[date] = (0 if idx == 0 else stock_holding_quantity.iloc[idx - 1])
                
            if date in VaR_historical.index:
                VaR_value.loc[date] = VaR_historical.loc[date]*portfolio_value.loc[date, "Portfolio Value"]
                CVaR_value.loc[date] = CVaR_historical.loc[date]*portfolio_value.loc[date, "Portfolio Value"]
            
        # remove all the rows before the first rebalancing date
        self.portfolio_value = portfolio_value[portfolio_value.index >= allocated_weights.index[0]]
        self.VaR_value = VaR_value[VaR_value.index >= allocated_weights.index[0]].ffill()
        self.CVaR_value = CVaR_value[CVaR_value.index >= allocated_weights.index[0]].ffill()
        
        # self.VaR_value = (VaR_value.ffill().mul(self.portfolio_value))[VaR_value.index >= allocated_weights.index[0]]
        # self.CVaR_value = (CVaR_value.ffill().mul(self.portfolio_value))[CVaR_value.index >= allocated_weights.index[0]]        
            
        
    def plot_portfolio_returns(self):
        """
        Plot the portfolio returns
        """
        plot_values = pd.concat([self.portfolio_value, self.VaR_value, self.CVaR_value], axis=1)
        plot_values.columns = ["Portfolio Value", "VaR", "CVaR"]
        plot_values.plot(figsize=(12, 8), title="Portfolio Value, VaR and CVaR", grid=True, ylabel="Value", xlabel="Date", legend=True)
        
    
    def calculate_alpha_beta(self):
        """
        Calculate the alpha and beta of the backtest results
        """
        
        benchmark_data = YahooDataSource("^GSPC", self.data_source.start_date, self.data_source.end_date)
        benchmark_data = benchmark_data.get_data_by_frequency(self.data_source.start_date, self.data_source.end_date, "1d")
        benchmark_data.ffill(inplace=True)
        benchmark_data = benchmark_data.loc[:, "^GSPC"].to_frame()
        benchmark_data = benchmark_data[benchmark_data.index >= self.portfolio_value.index[0]]
        benchmark_returns = benchmark_data.pct_change().dropna().values
        
        portfolio_returns = self.portfolio_value.pct_change().dropna().values
        
        x = sm.add_constant(benchmark_returns)
        y = portfolio_returns
        model = sm.OLS(y, x).fit()        
        
        return model.params[0], model.params[1]
        
    
    def get_summary(self):
        """
        Generate a summary of the backtest results including various statistics
        """
        
        portfolio_returns = self.portfolio_value.pct_change().dropna().values
        
        summary = {}
        summary["Total Return"] = self.portfolio_value.iloc[-1].values[0] - 1
        summary["Mean Daily Return"] = portfolio_returns.mean()
        summary["Std Dev of Daily Return"] = portfolio_returns.std()
        summary["Sharpe Ratio"] = summary["Mean Daily Return"] / summary["Std Dev of Daily Return"]
        
        cumulative_max = np.maximum.accumulate(self.portfolio_value)
    
        # Calculate the drawdown at each point in the array
        drawdowns = (self.portfolio_value - cumulative_max) / cumulative_max
        
        # Find the maximum drawdown
        summary["Max Drawdown"] = np.min(drawdowns)
        summary["Calmar Ratio"] = (summary["Total Return"] / abs(summary["Max Drawdown"]))
        
        summary["Alpha"], summary["Beta"] = self.calculate_alpha_beta()
        
        return summary
        