import pandas as pd
import os
from DataIngestion import DataIngestion as di
pd.set_option('display.max_rows', 200)


def strategy_1(path):
	"""
	Zack rank greater than  or equal to 3
	Most Accurate Est >= Current Qtr Est and EW_estimate >= Current Qtr Est
	"""
	trades = {'long': 'No long position', 'short': 'No short position'}
	
	useful_columns = ['tickers', 'z_rank', 'z_eps_diff', 'ew_eps_diff', 'market_cap',
					  'z_release_time', 'z_esp', 'z_industry', 'z_price',
					  'expected_date', 'position']
					  
	df = pd.read_csv(path)
	df['z_eps_diff'] = df['z_acc_est'] - df['z_curr_eps_est']
	df['ew_eps_diff'] = df['ew_eps'] - df['ew_curr_eps_est']
	trading_df = pd.DataFrame()

	strat1 = ((df['z_esp'] >= 0) &
			  (df['z_acc_est'] > df['z_curr_eps_est']) &
			  (df['ew_eps'] > df['ew_curr_eps_est']))

	strat2 = ((df['z_rank'] <= 3) &
			  (df['z_esp'] >= 0) &
			  (df['z_eps_diff'] > 0) &
			  (df['ew_eps_diff'] > 0))

	short_strat = ((df['ew_eps_diff'] < 0) &
				   (df['z_eps_diff'] < 0))

	for long_strats in [strat2, strat1]:
		long_df = df.loc[long_strats]
		long_df['position'] = 'Long'
		if not long_df.empty:
			trades['long'] = long_df[['tickers', 'expected_date', 'z_release_time']].to_dict(orient='records')

		short_df = df.loc[short_strat]
		short_df['position'] = 'Short'
		if not short_df.empty:
			trades['short'] = short_df[['tickers', 'expected_date', 'z_release_time']].to_dict(orient='records')
		
		trading_df = pd.concat([trading_df, long_df, short_df])
		if not trading_df.empty:
			break

	trading_df = trading_df[useful_columns]	
	trading_df = trading_df.sort_values(['market_cap'], ascending=False)
   
	amc = trading_df[trading_df['z_release_time'] == 'AMC']
	bmo = trading_df[trading_df['z_release_time'] == 'BMO']
	print('BMO DataFrame\n')
	print(bmo)
	print('\n')
	print('AMC DataFrame\n')
	print(amc) 
	print('\n')
	
	print(trades)
	#import pdb; pdb.set_trace()

def weekly_analysis(paths):
	cols = ['expected_date', 'consensus_eps', 'analysts', 'tickers', 'market_cap', 'ew_revenue', 'ew_curr_eps_est', 'ew_eps', 'z_esp', 'z_acc_est', 'z_curr_eps_est', 'z_release_time', 'z_rank', 'z_industry', 'z_price']
	industry_oi = ['Internet - Commerce', 'Computer - Software', 'Internet - Services', 'Internet - Software']
	df = pd.concat([pd.read_csv(i) for i in paths])
	df = df[cols]
	df['expected_date'] = df['expected_date'].map(pd.to_datetime)
	df = df.sort_values(['expected_date', 'market_cap'], ascending=True)
	df = df[df['market_cap'].notnull()]
	idf = df[df['z_industry'].isin(industry_oi)]
	import pdb; pdb.set_trace()


def get_zacks_ew_data(dates_dict):
	home_path = '/home/ksooklall/workspace/Stocks'
	paths = []
	# Set script to run once a day
	# create arg parser to input date (d) and strat(s) and get training df
	# exp: python Analysis.py -d '2018-Sep-01' -s 2 --> training_df
	
	for date, override in dates_dict.items():
	
		if not os.path.exists('{}/scraped_data/{}_ew_zack_df.pkl'.format(home_path, date)) or override:
			di_obj = di(tickers=[], date=date)
			path = di_obj.scrape_daily_df()
			paths.append(path)
		else:
			path = '{}/scraped_data/{}_ew_zack_df.pkl'.format(home_path, date)
			paths.append(path)

	for path in paths:	
		strategy_1(path)


def statistical_analysis():
	"""
	1 - Price to earnings (forwardPE): 
				What you pay for $1 of earning, compare to sector average
	2 - Price to book ratio (pbRatio): 
				Current share price / equity per share (book value)
				below 1 -> less than value of assets
				above 1 -> more than value of assets
	3 - Price to earning to growth (pegRatio):
				Below 1 -> Undervalued
	4 - Return on Equity (returnOnEquity):
				17-20%  very good 
				20-25%  excellent
				30%+	superior
	5 - Debt to equity ratio (debtToEquity):
				Total debt / shareholder equity
				The lower to better, avoid above 2
	6 - Current ratio 'liquidity ratio' (currentRatio):
				below 1 -> liabilities exceed assets
				above 1 -> assets exceed liabilities
				above 3 -> holding back
				best to be between 1 - 2
	"""
	
	bench_tickers = ['AMZN', 'TSLA', 'MAXR']

	rename_cols= { 'forward_p/e_1': 'forwardPE', 'price/book_(mrq)': 'pbRatio', 'peg_ratio_(5_yr_expected)_1': 'pegRatio', 'return_on_equity_(ttm)': 'returnOnEquity', 'total_debt/equity_(mrq)': 'debtToEquity', 'current_ratio_(mrq)': 'currentRatio', 'key_0': 'tickers'}
	
	data_in = di()
	ldf, hdf = data_in.get_hl52_stocks()
	tickers = ldf['symbol'].tolist() + hdf['symbol'].tolist() + bench_tickers
	data_in.set_tickers(tickers)
	raw_df = data_in.get_yahoo_statistics()

	# Long position
	long_cols = ['new_low', 'previous_low', 'high'] + list(rename_cols.keys())
	df = raw_df.merge(ldf[['new_low', 'previous_low', 'high']], left_on=raw_df.index, right_on=ldf['symbol'], how='inner')
	df = df[long_cols].rename(columns=rename_cols).set_index(['tickers'], drop=True)

	df['pct_to_gain'] = df['high'] / df['previous_low'] - 1
	df['pct_drop'] = (1 - df['previous_low'] / df['high'])
	df = df.sort_values(['pct_to_gain'])
	low_mask = df['returnOnEquity'] > 15
	medium_mask = low_mask & (df['pbRatio'] > 1)
	high_mask = medium_mask & (df['pegRatio'] < 1)
	ultra_mask = high_mask & (df['returnOnEquity'].between(17, 20))

	df[(df['forwardPE'] < 50) & (df['pbRatio'] > 1) & (df['pegRatio'] < 1) & (df['returnOnEquity'] > 17) & (df['debtToEquity'] < 2) & (df['currentRatio'].between(1, 2))]
	import pdb; pdb.set_trace()

if __name__ == '__main__':
	dates_dict = {'2020-06-08': True, '2020-06-09': True, '2020-06-10': True, '2020-06-11': True, '2020-06-12': True}
	get_zacks_ew_data(dates_dict)
	#statistical_analysis()


