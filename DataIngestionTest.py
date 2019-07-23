"""
Testing DataIngestion
"""
import numpy as np
import pandas as pd
from datetime import datetime
from DataIngestion import DataIngestion as di

TEST_TICKER = 'GOOGL'
TEST_DATE = '2019-May-31'

def test_get_whisper_numbers(klass):
	df = klass.get_whisper_numbers()
	assert isinstance(df, pd.DataFrame)
	assert np.all(df.columns == ['tickers', 'eps', 'ew_revenue', 'ew_curr_eps_est', 'ew_eps'])

def test_get_rsi_frame(klass):
	rsi = klass.get_rsi_frame()

def test_insider_trading(klass):
	df = klass.get_insider_trading('NDSN')
	test_df = pd.DataFrame()
	assert_frame_equal(df.head(), test_df.head())

	df = klass.get_insider_trading(['TWTR', 'SNAP'])
	test_df = pd.DataFrame()
	assert_frame_equal(df.head(), test_df.head())

def test_get_yahoo_statistics(klass):
	df = klass.get_yahoo_statistics()
	assert isinstance(df, pd.DataFrame)
	assert df.shape == (59, 1)


def test_get_earning_calender(klass):
	df = klass.get_earning_calender()
	assert isinstance(df, pd.DataFrame)
	assert np.all(df.columns == ['consensus_eps', 'analysts', 'EPS', 'suprise_pct', 
								 'tickers', 'mc_obj', 'market_cap', 'reported_date', 'name'])	

def test_get_hl52_stocks(klass):
	ldf, hdf = klass.get_hl52_stocks()
	assert isinstance(ldf, pd.DataFrame)
	assert isinstance(hdf, pd.DataFrame)

def test_get_zacks_numbers(klass):
	zdf = klass.get_zacks_numbers(tickers=TEST_TICKER)
	import pdb; pdb.set_trace()

if __name__ == '__main__':
	klass = di(tickers=[TEST_TICKER], date=TEST_DATE)
	#test_get_rsi_frame(klass)
	#test_insider_trading(klass)
	#test_get_hl52_stocks(klass)
	#test_get_yahoo_statistics(klass)	
	#test_get_earning_calender(klass)
	test_get_whisper_numbers(klass)
	#test_get_zacks_numbers(klass)
