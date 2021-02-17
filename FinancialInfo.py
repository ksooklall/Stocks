from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import swifter
import urllib3
import ssl
from bs4 import BeautifulSoup as bs

from PandasUtility import clean_columns
from yahoo_fin.stock_info import get_analysts_info, get_balance_sheet, get_cash_flow, get_day_gainers, get_day_losers, \
    get_stats_valuation, get_stats, get_holders, get_income_statement, get_quote_table, get_data, get_analysts_info
from yahoo_fin.options import get_calls, get_puts, get_options_chain, get_expiration_dates

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Nasdaq '^IXIC'
URLS = {'yahoo': 'https://finance.yahoo.com/calendar/earnings?day={}',
        'zack': 'https://www.zacks.com/stock/quote/{0}?q={0}'}


def get_stock_quotes(ticker):
    drop_cols = ['open', 'previous_close', 'ex-dividend_date', 'beta_(5y_monthly)', '1y_target_est', 'bid', 'ask']
    keep_cols = ['price', '52_week_range', "day's_range", 'eps_(ttm)', 'earnings_date',
                 'market_cap', 'pe_ratio_(ttm)', 'volume', 'avg._volume']

    df = pd.DataFrame(get_quote_table(ticker), index=[ticker])
    df = clean_columns(df).rename(columns={'quote_price': 'price'}).drop(drop_cols, axis=1)
    return df[keep_cols]


def get_stock_fundamentals(ticker):
    bs = get_balance_sheet(ticker).T
    bs.columns = bs.loc['Breakdown'].str.lower().str.replace(' ', '')
    bs = bs.drop('Breakdown', axis=0)

    cf = get_cash_flow(ticker)

    sv = get_stats_valuation(ticker) # NC
    ist = get_income_statement(ticker)


def get_stock_stats(ticker):
    ss = get_stats(ticker)
    sh = get_holders(ticker)    # NC


def get_earning_calendar(date):
    try:
        df_list = pd.read_html(URLS['yahoo'].format(date))
    except ValueError:
        return pd.DataFrame()

    df = df_list[0]
    if df.empty:
        return df

    df = df.loc[:, df.columns[:4]].rename(columns={'Symbol': 'tickers'})
    df = clean_columns(df)
    df['date'] = date

    df = df.sort_values(by='earnings_call_time').drop_duplicates(['tickers'], keep='first')
    return df


def get_earning_calendar_full(start, option_freq='weekly'):
    dates = get_dates(start)
    edf = pd.concat([get_earning_calendar(str(i.date())) for i in dates])\
        .sort_values(by=['date'])\
        .reset_index(drop=True)

    edf[['rsi', 'price']] = edf['tickers'].swifter.apply(get_rsi)
    edf['option_freq'] = edf['tickers'].swifter.apply(get_option_freq)
    edf = edf[edf['option_freq'].eq(option_freq) & edf['rsi'].gt(0)]

    return edf


def get_market_movers():
    gainers = get_day_gainers()
    losers = get_day_losers()


def get_rsi(stock, n=14, start_date='2020-01-01', return_num=True):
    try:
        df = get_data(stock, start_date=start_date)
    except AssertionError:
        return pd.Series({'rsi': -1.0, 'adjclose': 0})

    chg = df['adjclose'].diff()
    avg_gain = chg.mask(chg < 0, 0).ewm(com=n - 1, min_periods=n).mean()
    avg_loss = chg.mask(chg > 0, 0).ewm(com=n - 1, min_periods=n).mean()
    rs = (avg_gain / avg_loss).abs()
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = rsi

    if return_num:
        return df.iloc[-1][['rsi', 'adjclose']]
    return df


def get_dates(start):
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_day = 4 - start_date.weekday()
    end_date = str(start_date + timedelta(end_day))
    dates = pd.date_range(start=start_date, end=end_date)
    return dates


def get_option_freq(ticker):
    return 'weekly' if len(get_expiration_dates(ticker)) > 8 else 'monthly'


def main():
    stocks = ['MRNA', 'PFE', 'NVAX', 'SNY', 'GSK', 'INO', 'VBIV', 'AZN',
              'AMGN', 'ADPT', 'ALT', 'BNTX', 'CYDY', 'GILD', 'HTBX', 'REGN']

    get_option_freq('MU')
    edf = get_earning_calendar_full('2021-02-16')
    import pdb; pdb.set_trace()
    cores = cpu_count() - 1
    with Pool(cores) as p:
        df_lst = p.map(get_stock_quotes, stocks)
        df_rsi = p.map(get_rsi, stocks)

    dfl = pd.concat(df_lst)
    dfr = pd.concat(df_rsi).drop_duplicates('ticker', keep='last').set_index('ticker').filter(['rsi'])

    df = pd.merge(dfl, dfr, left_index=True, right_index=True).sort_values('rsi')
    print(df)
    df.head()


if __name__ == '__main__':
    main()
