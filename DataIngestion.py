"""
Class for collecting data
Given a list of tickers will use AlphaVantage to obtain data

Alpha Vantage:
raw_url - https://www.alphavantage.co/documentation/
generic_url - 'alphavantage/{}_tks.csv'.format(len(tickers))

Zacks Research Investment
raw_url - https://www.zacks.com/stock/quote/FB?q=fb
generic_url - https://www.zacks.com/stock/quote/FB?q=fb

EarningsWhispers
raw_url - https://www.earningswhispers.com/stocks/fb

Blue Chip Stocks
'https://www.nasdaq.com/screening/companies-by-industry.aspx?sortname=marketcap&sorttype=1&exchange=NASDAQ'
"""
from bs4 import BeautifulSoup as bs
from datetime import datetime
import pandas as pd
import requests


class DataIngestion():
    EW = 'https://www.earningswhispers.com/stocks/'
    
    def __init__(self, tickers, api_key=None):
        self.tickers = tickers
        self.api_key = api_key

    def collect_data(self, save=True, url=None):
        master_df = pd.DataFrame()
        for ticker in self.tickers:
            if not url:
                break
            url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&outputsize=full&apikey={}&datatype=csv'.format(ticker, self.api_key) 
            temp_df = pd.read_csv(url)
            temp_df['ticker'] = ticker
            master_df = master_df.append(temp_df)
            
        if save and url:
            master_df.to_csv(url)

    def scrape_data(self, url, df, save=True):
        try:
            df_list = pd.read_html(url)
            import pdb; pdb.set_trace()
        except ConnectionResetError:
            r = requests.get(url)
            
        def _get_whisper_numbers():
            r = requests.get(url + self.tickers[0])
            soup = bs(r.context, "html5lib")
            earnings_per_share = float(soup.find_all("div", class_="mainitem")[0].get_text().strip()[1:])
            consensus = float(soup.find_all("div", id="consensus")[0].get_text().strip()[-4:])
            revenue = float(soup.find_all("div", id="revest")[0].get_text().strip()[-4:])
            return earnings_per_share, consensus, revenue

        for stock in stocks:
            df.append(_get_whisper_numbers)
        
    def get_earning_calender(self):
        df_list = pd.read_html('https://www.nasdaq.com/earnings/earnings-calendar.aspx?date=2018-Aug-06')
        market_exp = {'M': 1e6, 'B': 1e9}
        rename = {'CompanyName(Symbol)MarketCapSortby:Name/Size': 'sym_mc_size',
                  'ExpectedReportDate': 'expected_date',
                  "LastYear'sReportDate": 'last_yr_report_date',
                  "LastYear'sEPS*": 'last_yr_eps',
                  'ConsensusEPS*Forecast': 'consensus_eps',
                  '%Suprise': 'suprise_pct',
                  '#ofEsts': 'analysts',
                  'FiscalQuarterEnding': 'quarter_ending'
                  }
        df = df_list[0]
        df.columns = df.columns.str.replace('\t', '').str.replace('\n', '').str.replace(' ', '')
        df = df.rename(columns=rename)
        import pdb; pdb.set_trace()

        df['ticker'] = df['sym_mc_size'].str.extract('.*\((.*)\).*')
        df['mc_obj'] = df['sym_mc_size'].str.split('$').str.get(1).str[:-1].astype(float)
        df['multiplier'] = df['sym_mc_size'].str.extract('\d+\.\d+(\w)')
        df['market_cap'] = df['mc_obj'].mul(df['multiplier'].map(market_exp))
        
        df = df[df['Symbol'].notnull()]
        return df
    
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
    #TODO: Organize urls and df_list in to a dict {df_list_index: url}
    api_key = 'XXXXXXXXXXXXXXXXX'
    tickers = ['FB']
    url = 'https://finance.yahoo.com/calendar/earnings?from=2018-07-22&to=2018-07-28&day=2018-07-25'
    di = DataIngestion(tickers, api_key)
    bcs = di.get_earning_calender()
    
