"""
Testing DataIngestion
"""
import numpy as np
import pandas as pd
from DataIngestion import DataIngestion as di



def test_get_rsi_frame(klass):
    rsi = klass.get_rsi_frame()

def test_insider_trading(klass):
    df = klass.get_insider_trading('NDSN')
    test_df = pd.DataFrame()
    assert_frame_equal(df.head(), test_df.head())

    df = klass.get_insider_trading(['TWTR', 'SNAP'])
    test_df = pd.DataFrame()
    assert_frame_equal(df.head(), test_df.head())

if __name__ == '__main__':
    klass = di(tickers=[], date=None)
    #test_get_rsi_frame(klass)
    test_insider_trading(klass)
    
