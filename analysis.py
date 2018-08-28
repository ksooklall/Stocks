from DataIngestion import DataIngestion
import pandas as pd
import os

def strategy_1(path, time='AMC'):
    """
    Zack rank greater than  or equal to 3
    Most Accurate Est >= Current Qtr Est and EW_estimate >= Current Qtr Est
    """

    df = pd.read_csv(path, encoding='ISO-8859-1')
    
    df = df[df['z_release_time'] == time]
    strat1 = ((df['z_rank'] <= 3) &
              (df['z_acc_est'] >= df['z_curr_eps_est']) &
              (df['ew_eps'] >= df['ew_curr_eps_est']))


    strat2 = ((df['z_rank'] <= 3) &
              (df['z_esp'] >= 0) &
              (df['z_acc_est'] >= df['z_curr_eps_est']) &
              (df['ew_eps'] >= df['ew_curr_eps_est']))
    
    strat3 = ((df['z_rank'] <= 3))
    
    trading_df = df.loc[strat2]
    if trading_df.empty:
        trading_df = df.loc[strat3]
        
    trading_df = trading_df[['tickers', 'z_rank', 'z_acc_est', 'z_curr_eps_est', 'ew_eps', 'ew_curr_eps_est',
                             'market_cap', 'z_release_time', 'z_esp', 'expected_date']]    
    trading_df = trading_df.sort_values(['z_rank', 'market_cap'])
    print(trading_df.head())
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    dates = ['2018-Aug-28', '2018-Aug-29', '2018-Aug-30', '2018-Aug-31']
    paths = []
    time = 'AMC'

    for date in dates:
        if not os.path.exists('scraped_data/{}_ew_zack_df.csv'.format(date)):
            di = DataIngestion(tickers=[], date=date)
            path = di.scrape_daily_df()
            paths.append(path)
        else:
            path = 'scraped_data/{}_ew_zack_df.csv'.format(date)
            paths.append(path)

    for path in paths:    
        strategy_1(path, time=time)
