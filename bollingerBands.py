""" Appling standard devitation analysis to stock prices"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.io.data import DataReader
import datetime
plt.style.use('ggplot')
# S&p500: ^GSPC

# Select trading range for analysis
start = datetime.datetime(2016,1,1)
end = datetime.datetime(2016,11,16)

# Select stock, window and band length
stock = 'FSLR'
window = 20
band_length = 2

# Collect data from yahoo
df = DataReader(stock,'yahoo',start)

# Calculate rolling mean of Adjust close
mean = pd.rolling_mean(df['Adj Close'],window)
# Calculate rolling standard devitation
std = pd.rolling_std(df['Adj Close'],window)
# Determine upper and lower bands
upperBand = mean+band_length*std
lowerBand = mean-band_length*std

# Buying analysis
if df['Adj Close'][-1]<lowerBand[-1]:
    print('Buy')
elif df['Adj Close'][-1]>upperBand[-1]:
    print('Sell')
else:
    print('Hold if owned')

# Calculate rolling mean of Daily Returns
df['Daily Returns'] = df['Adj Close']/df['Adj Close'].shift()-1
dmean = pd.rolling_mean(df['Daily Returns'],window)
dstd = pd.rolling_std(df['Daily Returns'],window)
dupperBand = dmean+band_length*dstd
dlowerBand = dmean-band_length*dstd

# Plotting
datemin = datetime.datetime(2016,start.month+1,1)
datemax = end
name = 'Adj Close'
plt.figure(figsize=(20,20))
plt.subplot(2,1,1)
plt.plot(df[name],color='k', ls = 'solid')
plt.plot(mean,color='r', label='Moving Average',linestyle = '-', linewidth = 3)
plt.plot(upperBand, color='b', label = 'Lower Band', linestyle = 'dotted', linewidth = 2)
plt.plot(lowerBand, color='b', label = 'Upper Band', linestyle = 'dotted', lw = 3)
plt.ylabel('Adj. Close')
plt.xlim(datemin,datemax)
plt.title(name)
plt.legend(loc=1)

plt.subplot(2,1,2)
plt.plot(df['Daily Returns'], color='k')
plt.plot(dmean,color='r', label='Moving Average',linestyle = '-', linewidth = 3)
plt.plot(dupperBand, color='b', label = 'Lower Band', linestyle = 'dotted', linewidth = 2)
plt.plot(dlowerBand, color='b', label = 'Upper Band', linestyle = 'dotted', lw = 3)
plt.xlim(datemin,datemax)
plt.ylabel('Adj. Close')
plt.title(name)
plt.show()
