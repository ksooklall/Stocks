import pandas as pd
import numpy as np
from pandas.io.data import DataReader
import datetime
import matplotlib.pyplot as plt

start = datetime.datetime(2016,3,21)
#S&P 500 ^GSPC

df = DataReader('FSLR','yahoo',start)
# Series of Adj Close
close = df['Adj Close'].dropna()
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
    change = np.diff(close)
    # Calculate for the first n days
    c = change[:n+1] # Inclusive exclusive operation
    up = c[c>0].sum()/n
    down = -c[c<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(close)
    rsi[:n]=100-100/(1+rs)

    # Calculate for n+1 -> today
    for i in range(n,len(close)):
        delta = change[i-1]
        if delta>0:
            newUp = delta
            newDown = 0 # Avoid referenced before assignment error
        else:
            newDown = -delta
            newUp = 0  # Avoid referenced before assignment error
        up = (up*(n-1)+newUp)/n # Find new mean with with new value
        down = (down*(n-1)+newDown)/n # Find new mean with with new value
        
        rsValue = up/down
        rsi[i]=(100-100/(1+rsValue))
    return rsi

""" Stochastic Oscillator(%K) follows the speed or the momentum of the price.
Momentum changes before the price changes
Measures the level of the closing price relative to low-high range over a peroid of time"""
def stochasticOscillator(close,n):
    # Base case: n days
    cArray = np.array(close)
    first = cArray[:n]
    baseHigh = highestN(first,n)
    baseLow = lowestN(first,n)
    stochasticOsc = np.zeros_like(close)
    stochasticOsc[:n] = (100)*(first-baseLow)/(baseHigh-baseLow)
    j = 1 # counter for the starting dates
    # nth cases: n+1 day -> today
    for i in range(n+1,len(close)):
        newHigh = highestN(cArray[j:i],n)
        newLow = lowestN(cArray[j:i],n)
        stochasticOsc[i] = (100)*(close[i]-baseLow)/(baseHigh-baseLow)
        j+=1
    return stochasticOsc

""" Williams %R rnages from -100 to 0.
%R>-20 => Sell signal
%R<-80 => Buy signal"""
def williamsR(close,n):
    # Base case: n days
    cArray = np.array(close)
    first = cArray[:n]
    baseHigh = highestN(first,n)
    baseLow = lowestN(first,n)
    williamR = np.zeros_like(close)
    williamR[:n] = (-100)*(baseHigh-first)/(baseHigh-baseLow)
    j = 1 # counter for the starting dates
    # nth cases: n+1 day -> today
    for i in range(n+1,len(close)):
        newHigh = highestN(cArray[j:i],n)
        newLow = lowestN(cArray[j:i],n)
        williamR[i] = (-100)*(newHigh-close[i])/(newHigh-newLow)
        j+=1
    return williamR
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

print('Williams %R: 14 - (-90.89)')
print(williamsR(close,14)[-5:])

df['stoOsc'] = stochasticOscillator(close,14)
df['wilR'] = williamsR(close,14)
macd, signalLine = movingAVGCD(close)
df['movAVGCD'] = macd
df['sigLine'] = signalLine
df['rsi'] = relativeStrengthIndex(close,14)

plt.figsize(10,7)
plt.subplot(3,1,1)
plt.plot(df['rsi'],ls = 'solid')
plt.ylabel('Relative Strength Indicator')
plt.title('Relative Strength Indicator')
plt.text('May 2016',50,df['rsi'].tail(1))
plt.axhline(20,color='r',ls='dashed',lw=2)
plt.axhline(70,color='r',ls='dashed',lw=2)
plt.legend(loc=2)

plt.subplot(3,1,3)
plt.plot(df['stoOsc'], ls='solid')
plt.ylabel('Stochastic oscillator')
plt.title('Stochastic Oscillator')

plt.subplot(3,1,2)
plt.plot(df['wilR'])
plt.ylabel('william %R')
plt.axhline(-20, color='r', ls='dashed')
plt.axhline(-80, color='r', ls='dashed')
plt.title('william %R')
plt.legend(loc=3)
plt.show()
