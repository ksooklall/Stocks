from DataIngestion import DataIngestion
import pandas as pd
import os

def strategy_1(path):
    """
    Zack rank greater than  or equal to 3
    Most Accurate Est >= Current Qtr Est and EW_estimate >= Current Qtr Est
    """
    trades = {'long': 'No long position', 'short': 'No short position'}
    
    useful_columns = ['tickers', 'z_rank', 'z_acc_est', 'z_curr_eps_est', 'ew_eps', 'ew_curr_eps_est',
                             'market_cap', 'z_release_time', 'z_esp', 'z_industry', 'z_price', 'expected_date', 'position']
    df = pd.read_csv(path, encoding='ISO-8859-1')
    trading_df = pd.DataFrame()

    strat1 = ((df['z_esp'] >= 0) &
              (df['z_acc_est'] > df['z_curr_eps_est']) &
              (df['ew_eps'] > df['ew_curr_eps_est']))

    strat2 = ((df['z_rank'] <= 3) &
              (df['z_esp'] >= 0) &
              (df['z_acc_est'] > df['z_curr_eps_est']) &
              (df['ew_eps'] > df['ew_curr_eps_est']))

    short_strat = (df['z_rank'] <= 3 &
                  (df['z_acc_est'] < df['z_curr_eps_est']) &
                  (df['ew_eps'] < df['ew_curr_eps_est']))

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
    trading_df = trading_df.sort_values(['z_rank', 'market_cap'])
    print(trading_df)
    print('\n')
    print(trades)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    dates_dict = {'2018-Oct-17': True, '2018-Oct-18': False, '2018-Oct-19': False}
    paths = []
    # Set script to run once a day
    # create arg parser to input date (d) and strat(s) and get training df
    # exp: python Analysis.py -d '2018-Sep-01' -s 2 --> training_df
    # can also input strat
    # add an option to override existing data (DONE)
    
    for date, override in dates_dict.items():
    
        if not os.path.exists('scraped_data/{}_ew_zack_df.csv'.format(date)) or override:
            di = DataIngestion(tickers=[], date=date)
            path = di.scrape_daily_df()
            paths.append(path)
        else:
            path = 'scraped_data/{}_ew_zack_df.csv'.format(date)
            paths.append(path)

    for path in paths:    
        strategy_1(path)
