from DataIngestion import DataIngestion
import pandas as pd
import os

def strategy_1(path):
    """
    Zack rank greater than  or equal to 3
    Most Accurate Est >= Current Qtr Est and EW_estimate >= Current Qtr Est
    """
    trades = {'long': 'No long position', 'short': 'No short position'}
    
    useful_columns = ['tickers', 'z_rank', 'z_eps_diff', 'ew_eps_diff', 'market_cap',
                      'z_release_time', 'z_esp', 'z_industry', 'z_price',
                      'expected_date', 'position']
                      
    df = pd.read_pickle(path)
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
    import pdb; pdb.set_trace()

def weekly_analysis(paths):
	cols = ['expected_date', 'consensus_eps', 'analysts', 'tickers', 'market_cap', 'ew_revenue', 'ew_curr_eps_est', 'ew_eps', 'z_esp', 'z_acc_est', 'z_curr_eps_est', 'z_release_time', 'z_rank', 'z_industry', 'z_price']
	industry_oi = ['Internet - Commerce', 'Computer - Software', 'Internet - Services', 'Internet - Software']
	df = pd.concat([pd.read_pickle(i) for i in paths])
	df = df[cols]
	df['expected_date'] = df['expected_date'].map(pd.to_datetime)
	df = df..sort_values(['expected_date', 'market_cap'], ascending=True)

	idf = df[df['z_industry'].isin(industry_oi)]
	import pdb; pdb.set_trace()

if __name__ == '__main__':
    dates_dict = {'2019-Apr-29': False, '2019-Apr-30': False, '2019-May-01': False, '2019-May-02': False, '2019-May-03': False}
    home_path = '/home/ksooklall/workspace/Stocks'
    paths = []
    # Set script to run once a day
    # create arg parser to input date (d) and strat(s) and get training df
    # exp: python Analysis.py -d '2018-Sep-01' -s 2 --> training_df
    # can also input strat
    # add an option to override existing data (DONE)
    
    for date, override in dates_dict.items():
    
        if not os.path.exists('{}/scraped_data/{}_ew_zack_df.pkl'.format(home_path, date)) or override:
            di = DataIngestion(tickers=[], date=date)
            path = di.scrape_daily_df()
            paths.append(path)
        else:
            path = '{}/scraped_data/{}_ew_zack_df.pkl'.format(home_path, date)
            paths.append(path)

    weekly_analysis(paths)

    for path in paths:    
        strategy_1(path)
