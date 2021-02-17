from _datetime import datetime
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

from yahoo_fin.stock_info import get_data, get_live_price

bv = ['NEPT', 'VFF', 'EGOV', 'OGI']
fg = ['NEPT', 'CL', 'IIPR'] # JUSH, CURA
mm = ['GRWG', 'TER', 'AMRS', 'MJ'] # 'JUSH','TURL'
other = ['TRUL', 'GTII', 'CGC']
rh = ['TLRY', 'EGOV', 'CRON']

remove = ['GTII']
lst = list(set(bv + fg + mm + rh + other) - set(remove))

start = '2020-07-01'
end = '2021-01-31'
today = str(datetime.now().date())

df = pd.concat([get_data(i, start_date=start, end_date=end)[['adjclose']].rename(columns={'adjclose': i}) for i in lst], axis=1)
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()

clean_weights = ef.clean_weights()
ef.portfolio_performance(verbose=True)

investment = 30000
last_prices = get_latest_prices(df)
new_weights = clean_weights

da = DiscreteAllocation(new_weights, last_prices, investment)
allocation, leftover = da.lp_portfolio()

fdf = pd.DataFrame({'ticker': allocation.keys(), 'shares': allocation.values()})
fdf['price'] = fdf['price'] = fdf['ticker'].map(get_live_price)
print(fdf)