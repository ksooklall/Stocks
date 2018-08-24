from DataIngestion import DataIngestion
import pandas as pd
import os

def strategy_1(path, time='AMC'):
    """
    Zack rank greater than  or equal to 3
    Most Accurate Est >= Current Qtr Est and EW_estimate >= Current Qtr Est
    """

    df = pd.read_csv(path, encoding='ISO-8859-1')
    strat1 = ((df['z_rank'] <= 3) &
              (df['z_acc_est'] >= df['z_curr_eps_est']) &
              (df['ew_eps'] >= df['ew_curr_eps_est']))


    strat2 = ((df['z_rank'] <= 3) &
              (df['z_esp'] >= 0) &
              (df['z_acc_est'] >= df['z_curr_eps_est']) &
              (df['ew_eps'] >= df['ew_curr_eps_est']))
    
    
    tickers = df.loc[strat2 & (df['z_release_time'] == time)]
    tickers = tickers[['ticker', 'z_rank', 'z_acc_est', 'z_curr_eps_est', 'ew_eps', 'ew_curr_eps_est', 'market_cap']]
    tickers = tickers.sort_values(['z_rank', 'market_cap'])
    
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    date = '2018-Aug-25'

    if not os.path.exists('scraped_data/{}_ew_zack_df.csv'.format(date)):
        di = DataIngestion(tickers=[], date=date)
        path = di.scrape_daily_df()
    else:
        path = 'scraped_data/{}_ew_zack_df.csv'.format(date)
        
    strategy_1(path, time='BMO')
