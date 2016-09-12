""" Technical analysis of the S&P500 index"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.io.data import DataReader
import datetime

# S&p500: ^GSPC

start = datetime.datetime(2016,1,1)
end = datetime.datetime(2016,9,1)
stock = 'FSLR'
df = DataReader(stock,'yahoo',start,end)

df['Date'] = df.index
df['Date'] = pd.to_datetime(df['Date'])
df['Date_delta'] = (df['Date']-df['Date'].min())/np.timedelta64(1,'D')

mean = pd.rolling_mean(df['Adj Close'],20)
std = pd.rolling_std(df['Adj Close'],20)
upperBand = mean+2*std
lowerBand = mean-2*std
"""
if df['Adj Close'][-1]<lowerBand[-1]:
    print('Buy')
elif df['Adj Close'][-1]>upperBand[-1]:
    print('Sell')
else:
    print('Hold if owned')"""

df['Daily Returns'] = df['Adj Close']/df['Adj Close'].shift()-1
dmean = pd.rolling_mean(df['Daily Returns'],20)
dstd = pd.rolling_std(df['Daily Returns'],20)
dupperBand = dmean+2*dstd
dlowerBand = dmean-2*dstd

# plotting
name = 'Adj Close'
plt.figure(figsize=(20,20))
plt.subplot(2,1,1)
plt.plot(df[name],color='k', ls = 'solid')
plt.plot(mean,color='r', label='Moving Average',linestyle = '-')
plt.plot(upperBand, color='b', label = 'Lower Band', linestyle = 'dotted', linewidth = 2)
plt.plot(lowerBand, color='b', label = 'Upper Band', linestyle = 'dotted', lw = 3)
plt.ylabel('Adj. Close')
plt.title(name)
plt.legend(loc=1)

plt.subplot(2,1,2)
plt.plot(df['Daily Returns'], color='k')
plt.plot(dmean,color='r', label='Moving Average',linestyle = '-')
plt.plot(dupperBand, color='b', label = 'Lower Band', linestyle = 'dotted', linewidth = 2)
plt.plot(dlowerBand, color='b', label = 'Upper Band', linestyle = 'dotted', lw = 3)
plt.ylabel('Adj. Close')
plt.title(name)
plt.legend(loc=2)
plt.show()
