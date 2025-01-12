from typing import Literal, List
import numpy as np
import pandas as pd
import yfinance as yf
import random
import warnings
from concurrent.futures import ThreadPoolExecutor

class YahooDataSource:

    def __init__(self, tickers, start_date, end_date, columns=["Close"], interval="1d"):
        self.columns = columns
        
        if type(tickers) == str:
            tickers = [tickers]
        if type(tickers) != list:
            raise ValueError("Tickers should be either a string or a list")
        
        if type(start_date) == str:
            start_date = pd.to_datetime(start_date)
        self.start_date = start_date
        
        if type(end_date) == str:
            end_date = pd.to_datetime(end_date)
        self.end_date = end_date
        
        self.interval = interval
        self.data = {}
        self.tickers = []
        self.add_tickers(tickers)

    def get_yahoo_data(self, ticker):
        try:
            # hist = yf.download(ticker, start=self.start_date, end=self.end_date, interval=self.interval)
            # hist["Return"] = hist[self.columns].pct_change()
            data = yf.Ticker(ticker)
            hist = data.history(
                start=self.start_date, end=self.end_date, interval=self.interval
            )
            hist["Return"] = hist["Close"].pct_change()
            hist = hist.dropna()
            # hist.reset_index(inplace=True)
            # if not hist.empty:
            #     for col in self.columns:
            #         data[symbol + "_" + col] = hist[col].to_numpy()
            return hist
        except Exception as e:
            raise e

    def add_ticker(self, ticker):
        print(f"Adding {ticker} to the data source")
        if ticker in self.tickers:
            warnings.warn(f"{ticker} already exists in the data source")
            return

        try:
            print(f"Getting data for {ticker}")
            self.tickers.append(ticker)
            data = self.get_yahoo_data(ticker)
            self.data[ticker] = data
            self.data[ticker].index = self.data[ticker].index.tz_convert(None)
        except Exception as e:
            warnings.warn(f"Failed to get data for {ticker}")
            self.tickers.remove(ticker)

    def add_tickers(self, tickers):
        with ThreadPoolExecutor() as executor:
            executor.map(self.add_ticker, tickers)

        # for ticker in tickers:
        #     self.add_ticker(ticker)

    def get_rand_returns(self, n=30):
        if n < 1:
            warnings.warn("Number of days should be greater than 0")
            return

        if n >= self.data[self.tickers[0]].shape[0]:
            warnings.warn("Number of days is greater than the available data")
            return

        if len(self.tickers) == 0:
            warnings.warn("No tickers available")
            return

        data = {}
        random_indexes = np.random.choice(
            self.data[self.tickers[0]].index, n, replace=False
        )
        for ticker in self.tickers:
            data[ticker] = self.data[ticker].loc[random_indexes, self.columns]

        # convert to dataframe
        data = pd.DataFrame(
            {key: data[key].values.flatten() for key in data}, index=random_indexes
        )

        return data

    def get_data_by_column_tickers(self, columns=-1, tickers=-1):

        all_tickers = self.tickers
        all_columns = self.columns

        if columns == -1:
            columns = all_columns

        if tickers == -1:
            tickers = all_tickers

        validated_tickers = set(tickers).intersection(all_tickers)
        validated_columns = set(columns).intersection(all_columns)

        if len(set(tickers)) != len(set(validated_tickers)):
            warnings.warn(
                f"Following Tickers are not Found {set(tickers)-set(validated_tickers)}"
            )

        if len(set(columns)) != len(set(validated_columns)):
            warnings.warn(
                f"Following Columns are not Found {set(columns)-set(validated_columns)}"
            )

        ticker_columns = self.create_ticker_columns(
            validated_columns, validated_tickers
        )

        return pd.DataFrame({key: self.data[key] for key in ticker_columns})

    def create_ticker_columns(self, columns, tickers):

        ticker_columns = []
        for tick in tickers:
            for col in columns:
                name = tick + "_" + col
                ticker_columns.append(name)

        return ticker_columns

    def get_tickers(self, ticker_columns):

        return [i.split("_")[0] for i in ticker_columns]

    def get_data(self, columns=-1):

        all_columns = self.columns

        if columns == -1:
            columns = all_columns

        validated_columns = list(set(columns).intersection(all_columns))

        if len(self.tickers) == 0:
            warnings.warn("No tickers available")
            return

        data = pd.DataFrame()

        for ticker in self.tickers:
            data[ticker] = self.data[ticker].loc[:, validated_columns]

        return data

    def get_data_by_frequency(
        self, start_date="", end_date="", frequency ="1d", columns: List = []
    ):
        
        if start_date == "":
            start_date = self.start_date
            
        if end_date == "":
            end_date = self.end_date
            
        # check if the start date is less than the end date and the start date is more than or equal the start date of the data source and the end date is less than or equal to the end date of the data source
        if start_date >= end_date:
            warnings.warn("Invalid start and end date")
            return
        
        if start_date < self.start_date:
            warnings.warn("Start date is less than the start date of the data source")
            return
        
        if end_date > self.end_date:
            warnings.warn("End date is greater than the end date of the data source")
            return
        
        if len(columns) == 0:
            columns = self.columns
            
        data = pd.DataFrame()
        for ticker in self.tickers:
            data[ticker] = self.data[ticker].loc[start_date:end_date, columns].resample(frequency).agg({"Close": "last"})
            # data[ticker] = self.data[ticker].loc[start_date:end_date, columns].asfreq(frequency, method="ffill", how="start")
            
        return data
        
        
