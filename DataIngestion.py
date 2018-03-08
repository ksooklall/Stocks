"""
Class for collecting data
Given a list of tickers will use AlphaVantage to obtain data
https://www.alphavantage.co/documentation/
"""
from datetime import datetime
import pandas as pd

class DataIngestion():
    def __init__(self, tickers, api_key):
        self.tickers = tickers
        self.api_key = api_key

    def collect_data(self, save=True, path=None):
        master_df = pd.DataFrame()
        for ticker in self.tickers:
            url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&outputsize=full&apikey={}&datatype=csv'.format(ticker, self.api_key) 
            temp_df = pd.read_csv(url)
            temp_df['ticker'] = ticker
            master_df = master_df.append(temp_df)
            
        if save and path:
            master_df.to_csv(path)

    def normalize():
        pass

    def standardize():
        pass

    def plot():
        pass

    def bol_bands():
        pass

    def moving_avg():
        pass

    def exp_avg():
        pass

    
if __name__ == '__main__':
    api_key = 'XXXXXXXXXXXXXXXXX'
    tickers = ['AMZN', 'GOOGL', 'MSFT']
    path = 'alphavantage/{}_tks.csv'.format(len(tickers))
    di = DataIngestion(tickers, api_key)
    di.collect_data(path=path)
