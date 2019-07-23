"""
Class for collecting data
Given a list of tickers will use AlphaVantage to obtain data

Alpha Vantage:
raw_url - https://www.alphavantage.co/documentation/
generic_url - 'alphavantage/{}_tks.csv'.format(len(tickers))

Zacks Research Investment:
raw_url - https://www.zacks.com/stock/quote/FB?q=fb
generic_url - https://www.zacks.com/stock/quote/FB?q=fb

EarningsWhispers:
raw_url - https://www.earningswhispers.com/stocks/fb

Blue Chip Stocks:
'https://www.nasdaq.com/screening/companies-by-industry.aspx?sortname=marketcap&sorttype=1&exchange=NASDAQ'

Earning calender:
https://www.nasdaq.com/earnings/earnings-calendar.aspx?date=2018-Aug-06

RSI:
https://www.stockmonitor.com/stock-screener/rsi-crossed-above-70/
Form4:
https://www.secform4.com/site/about.htm

Use 8/15/2018 as test, small amount of stocks
"""
from bs4 import BeautifulSoup as bs
from datetime import datetime
import pandas as pd
import requests
import urllib3
import ssl
from PandasUtility import preprocess_df, clean_columns, convert_float, convert_kmb_float
from ColumnRenames import form4, nasdaq
import re

class DataIngestion():
	# UNIFORM_RESOURCE_LOCATORS
	URLS = {
		'ew': 'https://www.earningswhispers.com/stocks/',
		'zacks': 'https://www.zacks.com/stock/quote/{0}?q={0}',
		'nasdaq': 'https://www.nasdaq.com/earnings/earnings-calendar.aspx?date={}',
		'rsi': 'https://www.stockmonitor.com/stock-screener/rsi-crossed-above-70/',
		'reuters_pg1': 'https://www.reuters.com/finance/stocks/insider-trading/{}.N',
		'reuters_pg2': 'https://www.reuters.com/finance/stocks/insider-trading/{}.N?symbol=&name=&pn=2&sortDir=&sortBy=',
		'sec_cik': 'http://www.sec.gov/cgi-bin/browse-edgar?CIK={}&Find=Search&owner=exclude&action=getcompany',
		'form4': 'https://www.secform4.com/insider-trading/{}.htm',
		'yahoo_statistics': 'https://finance.yahoo.com/quote/{0}/key-statistics?p={0}',
		'low_52': 'https://www.nasdaq.com/aspx/52-week-high-low.aspx?exchange=NASDAQ&status=LOW',
		'high_52': 'https://www.nasdaq.com/aspx/52-week-high-low.aspx?exchange=NASDAQ&status=HIGH'
		}

	MARKET_EXP = {'M': 1e6, 'B': 1e9}

	def __init__(self, date=None, tickers=None):
		# Date format: Y-m-d, dtype:str ex: '2018-Aug-06'
		self.tickers = tickers
		self.date = date


	def set_tickers(self, tickers):
		"""
		Set tickers attribute
		"""
		self.tickers = tickers


	def check_df_type(self, df):
		"""
		Check dtype of df and set it if it doesn't exist
		"""
		if df is None:
			df = pd.DataFrame({'tickers': self.tickers})
		return df


	def get_earning_calender(self):
		"""
		Scrape nasdaq for all stocks that will be releaseing Q4
		"""
		nasdaq_drop_cols = ['quarter_ending', 'multiplier', 'ReportedDate', 'Time', 'sym_mc_size']
		numeric_cols = ['z_rank', 'z_acc_est', 'z_curr_eps_est', 'ew_eps', 'ew_curr_eps_est', 'z_esp']
		df_list = pd.read_html(self.URLS['nasdaq'].format(self.date))

		df = df_list[0]
		if df.empty:
			return df
		df.columns = df.columns.str.replace('\t', '').str.replace('\n', '').str.replace(' ', '')
		df = df.rename(columns=nasdaq)

		# If only one report, there won't be an extra header file (Need example)
		# Not sure why this logic is here (Investigate)
		if df[df.columns[0]].iloc[0] == 'Time':
			df = df.loc[1:].reset_index(drop=True)
		else:
			df = df.reset_index(drop=True)

		df['tickers'] = df['sym_mc_size'].str.extract('.*\((.*)\).*', expand=False)
		df['mc_obj'] = df['sym_mc_size'].str.split('$').str.get(1).str[:-1].astype(float)
		df['multiplier'] = df['sym_mc_size'].str.extract('\d+\.\d+(\w)', expand=False)
		df['market_cap'] = df['mc_obj'].mul(df['multiplier'].map(self.MARKET_EXP))

		# Cleaning up the dataframe
		df = df[df['tickers'].notnull()]
		df = convert_float(df, columns=['consensus_eps', 'EPS'])
		df['reported_date'] = df['ReportedDate'].apply(pd.to_datetime, dayfirst=True)
		df['name'] = df['sym_mc_size'].str.extract('([^(:]+)')
		df = df.drop(nasdaq_drop_cols, axis=1)
		df = df.sort_values(['market_cap'], ascending=False)

		print('Completed nasdaq scraping')
		self.set_tickers(df['tickers'].tolist())
		return df

	def get_whisper_numbers(self, df=None):
		print('Starting earning whisper scraping')
		ew_drop_cols = ['multiplier', 'eps_consensus_revenue', 'rev1', 'consensus', 'revenue']
		numeric_cols = ['ew_eps', 'ew_curr_eps_est', 'eps']

		df = self.check_df_type(df)

		def scrape_whisper_numbers(ticker):
			r = requests.get(self.URLS['ew'] + ticker)
			soup = bs(r.text, "html5lib")
			soup_eps = soup.find_all("div", class_='mainitem')
			
			if not soup_eps:
				return [None, None, None]
			earnings_per_share = soup_eps[0].get_text().strip()
			consensus = soup.find_all("div", id="consensus")[0].get_text().strip()
			revenue = soup.find_all("div", id="revest")[0].get_text().strip()
			return [earnings_per_share, consensus, revenue]

		# EW cleaning
		df['eps_consensus_revenue'] = df['tickers'].map(scrape_whisper_numbers)
		df[['eps', 'consensus', 'revenue']] = pd.DataFrame(df['eps_consensus_revenue'].values.tolist(), index=df.index)
		df[['rev1', 'multiplier']] = df['revenue'].str.extract(r'(\d+\.\d+ (\w))')[0].str.split(' ', n=1, expand=True)
		df['ew_revenue'] = (df['rev1'].astype(float) * df['multiplier'].map(self.MARKET_EXP)).fillna(-1)
		df['ew_curr_eps_est'] = df['consensus'].replace('[\$,)]','', regex=True).replace('[(]','-', regex=True).replace('(Consensus: *)', '', regex=True).astype(float)
		df['ew_eps'] = df['eps'].replace('[$[)]','', regex=True).replace('[(]','-', regex=True)
		df = df.drop(ew_drop_cols, axis=1)
		df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
		print('Completed earning whisper scraping')
		return df


	def get_zacks_numbers(self, df=None):
		print('Starting zacks research scraping')
		z_drop_cols = ['zacks']
		z_numeric_cols = ['z_rank', 'z_acc_est', 'z_curr_eps_est', 'z_esp']

		df = self.check_df_type(df)			
		def scrape_zacks_information(ticker):
			# Figure out how to handle with pd.read_html()
			# ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host	
			ctx = ssl.create_default_context()
			ctx.check_hostname = False
			ctx.verify_mode = ssl.CERT_NONE

			http = urllib3.PoolManager()
			res = http.request('GET', self.URLS['zacks'].format(ticker))
			soup = bs(res.data.decode('utf-8'))
			
			tables = soup.find_all('table')
			if len(tables) <=5:
				return [None] * 14

			industry = soup.find_all(class_='sector')[0].get_text()
			price = soup.find_all(class_='last_price')[0].get_text()
			
			# Get ESP, Accurate EST, Earning ESP, Current Qtr Est, Report Release Time, Forward PE, PEG Ratio 
			esp_df = pd.read_html(str(tables[3]))[0]
			esp, acc_est, curr_eps_est, _, earning_date, _, _, forward_pe, peg_ratio = esp_df[1].values
			# Get z_rank, ind_rank, sector_rank, value, growth, momentum, vgm
			rank_df = pd.read_html(str(tables[5]))[0]
			z_rank, ind_rank, sector_rank, _, _, _ = rank_df[1].values
			value, growth, momentum, vgm = rank_df.loc[3][0].split('|')

			# Optimize below to happen in df above or better
			value, growth, momentum, vgm = value[-8:-7], growth[1], momentum[1], vgm[1]		
			return [esp, acc_est, curr_eps_est, earning_date, forward_pe, peg_ratio, z_rank, ind_rank, sector_rank, growth, momentum, vgm, industry, price]

		# Zack cleaning
		df['zacks'] = df['tickers'].map(scrape_zacks_information)
		df[['z_esp', 'z_acc_est', 'z_curr_eps_est', 'z_release_time', 'z_forward_pe', 'z_peg_ratio', 'z_rank', 'z_ind_rank', 'z_sector_rank', 'z_growth', 'z_momentum', 'z_vgm', 'z_industry', 'z_price']] = pd.DataFrame(df['zacks'].values.tolist(), index=df.index)
		df = df.dropna(subset=['z_rank'])
		df['z_rank'] = df['z_rank'].str.get(-1)
		df['z_release_time'] = df['z_release_time'].str.extract(r'([A-Z]+)', expand=False)
		df['z_esp'] = df['z_esp'].str.replace('%', '')
		df['z_price'] = df['z_price'].str.extract(r'(\d+\.\d+\d+)', expand=False)
		df['z_industry'] = df['z_industry'].str.replace('Industry: ', '')
		df = df.drop(z_drop_cols, axis=1)
		df[z_numeric_cols] = df[z_numeric_cols].apply(pd.to_numeric, errors='coerce')

		print('Completed zacks scraping')
		return df	


	def get_yahoo_statistics(self):
		"""
		 Scrapes yahoo finance statistical page based on the list of tickers
		"""
		df = pd.DataFrame()		
		for ticker in self.tickers:
			url = 'https://finance.yahoo.com/quote/{0}/key-statistics?p={0}'.format(ticker)
			ldf = pd.read_html(url)
			tdf = pd.concat([i for i in ldf]).set_index([0], drop=True).rename(columns={1: ticker}).T
			df = pd.concat([df, tdf], sort=False)
		
		df = clean_columns(df)
		df[['52_low', '52_high']] = df['52_week_range'].str.replace(' ', '').str.split('-', expand=True).astype(float)
		df = df.drop(['52_week_range'], axis=1)

		pct_cols = ['%_held_by_insiders_1', '%_held_by_institutions_1', '52-week_change_3', 'return_on_assets_(ttm)', 'return_on_equity_(ttm)', 's&p500_52-week_change_3', 'short_%_of_shares_outstanding_(apr_30,_2019)_4', 'payout_ratio_4', 'operating_margin_(ttm)', 'profit_margin', 'short_%_of_float_(apr_30,_2019)_4', 'forward_annual_dividend_yield_4', 'trailing_annual_dividend_yield_3', 'payout_ratio_4', 'expense_ratio_(net)', 'quarterly_earnings_growth_(yoy)', 'yield', 'ytd_return', 'quarterly_revenue_growth_(yoy)']

		kmb_cols = ['market_cap_(intraday)_5', 'enterprise_value_3', 'revenue_(ttm)', 'gross_profit_(ttm)', 'ebitda', 'net_income_avi_to_common_(ttm)', 'total_cash_(mrq)', 'total_debt_(mrq)', 'operating_cash_flow_(ttm)', 'levered_free_cash_flow_(ttm)', 'avg_vol_(3_month)_3', 'avg_vol_(10_day)_3', 'shares_outstanding_5', 'float', 'shares_short_(apr_30,_2019)_4', 'shares_short_(prior_month_mar_29,_2019)_4', 'net_assets']

		date_cols = ['fiscal_year_ends', 'most_recent_quarter_(mrq)', 'last_split_date_3', 'dividend_date_3', 'ex-dividend_date_4', 'inception_date']

		unused = ['bid', 'ask', 'last_split_factor_(new_per_old)_2', "day's_range"]
		
		float_cols = list(set(df.columns)-set(kmb_cols)-set(pct_cols)-set(date_cols)-set(unused))
		df = convert_float(df, pct_cols + float_cols, to_replace='%')
		df = convert_kmb_float(df, kmb_cols)
		return df


	def get_hl52_stocks(self):
		"""
		Scrapes nasdaq page for stocks that hit 52 week low/high
		"""
		print('Getting 52 week low stocks ....')
		ldf = pd.read_html('https://www.nasdaq.com/aspx/52-week-high-low.aspx?exchange=NASDAQ&status=LOW')[0]
		ldf = ldf[ldf['Symbol'].str.len() < 5]
		print('Getting 52 week high stocks ....')
		hdf = pd.read_html('https://www.nasdaq.com/aspx/52-week-high-low.aspx?exchange=NASDAQ&status=HIGH')[0	]
		hdf = hdf[hdf['Symbol'].str.len() < 5]
 
		ldf = preprocess_df(ldf, float_cols=['new_low', 'previous_low', 'high'])
		hdf = preprocess_df(hdf, float_cols=['new_high', 'previous_high', 'previous_low'])

		return [ldf, hdf]


	def get_rsi_frame(self, price_range=[10, 100]):
		lower, upper = price_range
		df_list = pd.read_html(self.URLS['rsi'])
		df = df_list[1]
		df.columns = df.columns.str.replace('\n', '').str.replace(' ', '').str.replace('[^a-zA-Z]', '').str.lower()
		df = df.query('bid > 0 and ask > 0')
		df = df.drop(['high', 'low'], axis=1)
		df['price'] = df['price'].str.extract('(\d+\.\d+)', expand=False)
		df['change'] = df['change'].str.extract('.*\((.*)\).*', expand=False).str.extract('(\d+\.\d+)')
		df[['price', 'change']] = df[['price', 'change']].apply(pd.to_numeric, errors='coerce')
		
		df = df.loc[df['price'].between(lower, upper)]
		df = df.sort_values(['volume'])


	def get_insider_trading(self, ticker):
		"""
		Scrapes secform4.com for insider trading information
		"""
		# List of common insider positions
		lst = ['CEO', 'VP', 'CFO', 'Director']
		
		if isinstance(ticker, list):
			ticker_lst = ticker
		else:
			ticker_lst = [ticker]
			
		df = pd.DataFrame()
		cik_lst = {i: self.get_cik_number(i) for i in ticker_lst}
		
		for tkr, cik in cik_lst.items():
			sdf = pd.read_html(self.URLS['form4'].format(cik))
			
			sdf[2] = sdf[2].drop(['TotalAmount'], axis=1)
			sdf[3] = sdf[3].drop(['ExercisableExpiration', 'Symnbol'], axis=1).rename(
				columns={'ConversionPrice': 'AveragePrice'})
			
			sdf = pd.concat([sdf[2], sdf[3]]).drop(['ReportedDateTime', 'Filing'], axis=1)
			sdf = clean_columns(sdf)

			sdf['tran_type'] = sdf['transactiondate'].str.replace(pat=r'(\d+-\d+-\d+)', repl='')
			sdf['transactiondate'] = pd.to_datetime(sdf['transactiondate'].str.extract(r'(\d+-\d+-\d+)'))
			sdf['shares_type'] = sdf['sharesowned'].str.replace(r'[^(A-Za-z^)]', '')
			sdf['sharesowned'] = sdf['sharesowned'].str.replace(r'[(A-Za-z)]', '')
			
			sdf = convert_float(sdf, ['averageprice', 'sharestraded', 'sharesowned'])
			sdf['symbol'] = sdf['symbol'].ffill()
			sdf['cik'] = cik
			
			sdf = sdf.rename(columns=form4).sort_values(['date'], ascending=False)
			sdf['insider_name'] = sdf['insider_pos'].str.replace('(' + '|'.join(lst)+')', '')
			sdf['insider_pos'] = sdf['insider_pos'].str.extract('(' + '|'.join(lst)+')', expand=False)
			
			df = pd.concat(df, sdf)

		return df

	def get_cik_number(self, ticker):
		"""
		The Central Index Key (CIK) is used on the SEC's computer systems to identify corporations
		and individual people who have filed disclosure with the SEC.
		"""
		cik_re = re.compile(r'.*CIK=(\d{10}).*')
		cik = cik_re.findall(requests.get(self.URLS['sec_cik'].format(ticker), stream=True).text)
		if len(cik):
			# Remove trailing 0s
			cik[0] = int(re.sub('\.[0]*', '.', cik[0]))
			return cik[0]
		

	def scrape_daily_df(self):
		print("Begin scraping for {}...".format(self.date))
		df = self.get_earning_calender()
		path = 'scraped_data/{}_ew_zack_df.pkl'.format(self.date)
		df.to_pickle(path)
		print("Completed scraping .... data located in {}".format(path))
		return path
