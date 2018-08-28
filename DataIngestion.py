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

Earning calender
https://www.nasdaq.com/earnings/earnings-calendar.aspx?date=2018-Aug-06

Use 8/15/2018 as test, small amount of stocks
"""
from bs4 import BeautifulSoup as bs
from datetime import datetime
import pandas as pd
import requests
import urllib3
import ssl

class DataIngestion():
    # UNIFORM_RESOURCE_LOCATORS
    URLS = {
        'ew': 'https://www.earningswhispers.com/stocks/',
        'zacks': 'https://www.zacks.com/stock/quote/{0}?q={0}',
        'nasdaq': 'https://www.nasdaq.com/earnings/earnings-calendar.aspx?date={}'
        }
    
    def __init__(self, tickers, date, api_key=None):
        # Date format: Y-m-d, dtype:str ex: '2018-Aug-06'
        self.tickers = tickers
        self.api_key = api_key
        self.date = date

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
       
    def get_earning_calender(self):
        market_exp = {'M': 1e6, 'B': 1e9}
        ew_drop_cols = ['sym_mc_size', 'Time', 'multiplier', 'mc_obj', 'eps_consensus_revenue', 'rev1', 'eps', 'consensus']
        z_drop_cols = ['zacks']
        nasdaq_drop_cols = ['quarter_ending']
        numeric_cols = ['z_rank', 'z_acc_est', 'z_curr_eps_est', 'ew_eps', 'ew_curr_eps_est']
        
        rename = {'CompanyName(Symbol)MarketCapSortby:Name/Size': 'sym_mc_size',
                  'ExpectedReportDate': 'expected_date',
                  "LastYear'sReportDate": 'last_yr_report_date',
                  "LastYear'sEPS*": 'last_yr_eps',
                  'ConsensusEPS*Forecast': 'consensus_eps',
                  '%Suprise': 'suprise_pct',
                  '#ofEsts': 'analysts',
                  'FiscalQuarterEnding': 'quarter_ending'}

        df_list = pd.read_html(self.URLS['nasdaq'].format(self.date))
        df = df_list[0]
        df.columns = df.columns.str.replace('\t', '').str.replace('\n', '').str.replace(' ', '')
        df = df.rename(columns=rename)

        # If only one report, there won't be an extra header file
        if len(df) == 1:
            df = df.reset_index(drop=True)
        else:
            df = df.loc[1:].reset_index(drop=True)
        
        df['tickers'] = df['sym_mc_size'].str.extract('.*\((.*)\).*', expand=False)
        df['mc_obj'] = df['sym_mc_size'].str.split('$').str.get(1).str[:-1].astype(float)
        df['multiplier'] = df['sym_mc_size'].str.extract('\d+\.\d+(\w)', expand=False)
        df['market_cap'] = df['mc_obj'].mul(df['multiplier'].map(market_exp))
        df = df.drop(nasdaq_drop_cols, axis=1)
        df = df[df['tickers'].notnull()] 
        print('Completed nasdaq scraping')

        # EW cleaning
        df['eps_consensus_revenue'] = df['tickers'].map(self.get_whisper_numbers)
        df[['eps', 'consensus', 'revenue']] = pd.DataFrame(df['eps_consensus_revenue'].values.tolist(), index=df.index)
        df[['rev1', 'multiplier']] = df['revenue'].str.extract(r'(\d+\.\d+ (\w))')[0].str.split(' ', n=1, expand=True)
        df['ew_revenue'] = (df['rev1'].astype(float) * df['multiplier'].map(market_exp)).fillna(-1)
        df['ew_curr_eps_est'] = df['consensus'].replace('[\$,)]','', regex=True).replace('[(]','-', regex=True).replace('(Consensus: *)', '', regex=True).astype(float)
        df['ew_eps'] = df['eps'].replace('[$[)]','', regex=True).replace('[(]','-', regex=True)
        df = df.drop(ew_drop_cols, axis=1)
        print('Completed ew scraping')
        
        # Zack cleaning
        df['zacks'] = df['tickers'].map(self.get_zacks_numbers)
        df[['z_esp', 'z_acc_est', 'z_curr_eps_est', 'z_release_time', 'z_forward_pe', 'z_peg_ratio', 'z_rank', 'z_ind_rank', 'z_sector_rank', 'z_growth', 'z_momentum', 'z_vgm']] = pd.DataFrame(df['zacks'].values.tolist(), index=df.index)
        df = df.dropna(subset=['z_rank'])
        df['z_rank'] = df['z_rank'].str.get(-1)
        df['z_release_time'] = df['z_release_time'].str.extract(r'([A-Z]+)', expand=False)
        df['z_esp'] = df['z_esp'].str.replace('%', '')
        df = df.drop(z_drop_cols, axis=1)
        print('Completed zacks scraping')

        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        return df

    def get_whisper_numbers(self, ticker):
        r = requests.get(self.URLS['ew'] + ticker)
        soup = bs(r.text, "html5lib")
        earnings_per_share = soup.find_all("div", class_="mainitem")[0].get_text().strip()
        consensus = soup.find_all("div", id="consensus")[0].get_text().strip()
        revenue = soup.find_all("div", id="revest")[0].get_text().strip()
        return [earnings_per_share, consensus, revenue]

    def get_zacks_numbers(self, ticker):
        # Figure out how to handle with pd.read_html()
        # ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host
        
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        http = urllib3.PoolManager()
        res = http.request('GET', self.URLS['zacks'].format(ticker))
        soup = bs(res.data.decode('utf-8'))

        tables = soup.find_all('table')
        # Get ESP, Accurate EST, Earning ESP, Current Qtr Est, Report Release Time, Forward PE, PEG Ratio 
        esp_df = pd.read_html(str(tables[3]))[0]
        esp, acc_est, curr_eps_est, _, earning_date, _, _, forward_pe, peg_ratio = esp_df[1].values
        
        # Get z_rank, ind_rank, sector_rank, value, growth, momentum, vgm
        rank_df = pd.read_html(str(tables[5]))[0]
        z_rank, ind_rank, sector_rank, _, _, _ = rank_df[1].values

        value, growth, momentum, vgm = rank_df.loc[3][0].split('|')

        # Optimize below to happen in df above or better
        value, growth, momentum, vgm = value[-8:-7], growth[1], momentum[1], vgm[1]        
        return [esp, acc_est, curr_eps_est, earning_date, forward_pe, peg_ratio, z_rank, ind_rank, sector_rank, growth, momentum, vgm]
                           
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

    def scrape_daily_df(self):
        print("Begin scraping for {}...".format(self.date))
        df = self.get_earning_calender()
        path = 'scraped_data/{}_ew_zack_df.csv'.format(self.date)
        df.to_csv(path)
        print("Completed scraping .... data located in {}".format(path))
        return path
