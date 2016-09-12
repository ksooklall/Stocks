import pandas as pd
import numpy as np
from pandas.io.data import DataReader
import datetime
import matplotlib.pyplot as plt

start = datetime.datetime(2016,3,21)
#end = datetime.datetime(2016,9,9)
df = DataReader('AMZN','yahoo',start)
# Series of Adj Close
close = df['Adj Close']
# Compute daily return
dailyReturn = df['Adj Close']/df['Adj Close'].shift(1)-1

# Highest price over n days
def highestN(close, n):
    return close[-n:].max()
# Lowest price over n days
def lowestN(close, n):
    return close[-n:].min()

# Simple(SMA) and Exponential moving average(EMA)
def movingAverage(close,n,type='Simple'):
    arr = np.asarray(close)
    if type=='Simple':
        return pd.rolling_mean(close, n)
    else:
        s = arr
        for t in range(n,len(close)):
            s[t] = (1/n)*arr[t]+((n-1)/n)*s[t-1]
        return pd.Series(s)
    
""" Relative Strength Index(RSI) is a momentum indicator to determine overbought or over sold
RSI = 100*(1-1/(1+RS), RS = mean(gain in n days)/mean(loss in n days)
RSI >70 => Stock Overbought good time to sell
RSI <30 => Stock Oversold good time to buy"""
def relativeStrengthIndex(close,n):
    deltas = np.diff(close)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    return 100-100/(1+rs)

""" Stochastic Oscillator(%K) follows the speed or the momentum of the price.
Momentum changes before the price changes
Measures the level of the closing price relative to low-high range over a peroid of time"""
def stochasticOscillator(close,n):
    return 100*(close-lowestN(close,n))/(highestN(close,n)-lowestN(close,n))

""" Williams %R rnages from -100 to 0.
%R>-20 => Sell signal
%R<-80 => Buy signal"""
def williamsR(close,n):
    return (highestN(close,n)-close)/(highestN(close,n)-lowestN(close,n))*(-100)

""" On balance volume is used to find buying and selling trends """
def onBalanceVolume(close):
    return null

""" Moving average convergence divergence(MACD)
MACD < signalLine => sell
MACD > signalLine => buy
"""
def movingAVGCD(close):
    macd = movingAverage(close,12)-movingAverage(close,26)
    signalLine = movingAverage(macd,9)
    return macd, signalLine

df['stoOsc'] = stochasticOscillator(close,14)
df['wilR'] = williamsR(close,14)
macd, signalLine = movingAVGCD(close)
df['movAVGCD'] = macd
df['sigLine'] = signalLine
rsi = relativeStrengthIndex(close,14)

plt.subplot(2,1,1)
plt.plot(df['stoOsc'],ls = 'solid')
plt.ylabel('Stochastic Oscillator')
plt.title('Stochastic Oscillator')
plt.legend(loc=4)

plt.subplot(2,1,2)
plt.plot(df['wilR'])
plt.ylabel('william %R')
plt.axhline(-20, color='r', ls='dashed')
plt.axhline(-80, color='r', ls='dashed')
plt.title('william %R')
plt.legend(loc=4)
plt.show()
