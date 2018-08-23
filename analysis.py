import pandas as pd

def strategy_1(time='AMC'):
    """
    Zack rank greater than  or equal to 3
    Most Accurate Est >= Current Qtr Est and EW_estimate >= Current Qtr Est
    """
    df = pd.read_csv('ew_zack_df.csv', encoding='ISO-8859-1')
    strat1 = ((df['z_rank'] <= 3) &
              (df['z_acc_est'] >= df['z_curr_eps_est']) &
              (df['ew_eps'] >= df['ew_curr_eps_est']))


    strat2 = ((df['z_rank'] <= 3) &
              (df['z_esp'] >= 0) &
              (df['z_acc_est'] >= df['z_curr_eps_est']) &
              (df['ew_eps'] >= df['ew_curr_eps_est']))
    
    
    tickers = df.loc[strat2 & (df['z_release_time'] == time)]
    tickers = tickers.sort_values(['z_rank', 'market_cap'])
    
strategy_1()
