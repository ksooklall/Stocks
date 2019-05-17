"""
Testing DataIngestion
"""
import numpy as np
import pandas as pd
from datetime import datetime
from DataIngestion import DataIngestion as di

TEST_TICKER = 'GOOGL'
TEST_DATE = '2019-Apr-15'

def test_whisper_numbers(klass):
	earnings_per_share, consensus, revenue = klass.get_whisper_numbers(TEST_TICKER)
	assert earnings_per_share
	assert consensus
	assert revenue


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
	assert np.all(df.columns == ['Time', 'sym_mc_size', 'ReportedDate', 'consensus_eps', 'analysts',
       'EPS', 'suprise_pct', 'tickers', 'mc_obj', 'multiplier', 'market_cap'])	


def test_get_hl52_stocks(klass):
	ldf, hdf = klass.get_hl52_stocks()
	assert isinstance(ldf, pd.DataFrame)
	assert isinstance(hdf, pd.DataFrame)


if __name__ == '__main__':
	klass = di(tickers=[TEST_TICKER], date=TEST_DATE)
	#test_get_rsi_frame(klass)
	#test_insider_trading(klass)
	test_whisper_numbers(klass)
	test_get_hl52_stocks(klass)
	test_get_yahoo_statistics(klass)	
	test_get_earning_calender(klass)
