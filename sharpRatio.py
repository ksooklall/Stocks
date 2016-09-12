"""
@author: ksook

Sharpe ratio and other portfolio statistics
"""

import pandas as pd
import numpy as np
from pandas.io.data import DataReader
import datetime
import scipy as sp
import scipy.optimize as scopt
import scipy.stats as spstats
import matplotlib.pyplot as plt


def getHistoricCloses(stock, startDate, endDate):
    # Obtain historic data
    df = DataReader(stock,'yahoo',startDate, endDate)
    if stock=='^GSPC':
        stock = 'SPY'
    df = df.rename(columns={'Adj Close':stock})
    return df[stock]

""" The daily return of a stock
can also use: closePrice.pct_change()
@param - DataFrame of closing prices
@return - DataFrame of % change"""                
def dailyReturns(closePrice):
    return closePrice/closePrice.shift(1)-1

""" Cumulative return
@param - current portfolio values
@return - float
"""
def cumReturns(portValue):
    return portValue[-1]/portValue[0]-1    

""" Normalize the data
Nor = Closing_Price[t]/ClosingPrice[start]
@param - DataFrame
@return - DataFrame"""
def normalizingData(df):
    return df/df.ix[0]

""" Sharpe ratio - Credit to William F. Sharpe
Risk adjusted return, all else being equal
    lower risk is better
    higher return is better
Sr = k*(Portfolio return(Rp)-Risk free rate(Rf))/ StdDev of portfolio return (Rstd)
     k - sqrt(# samples per year)
     k = sqrt(252) => daily
     k = sqrt(52) => weekly
     k = sqrt(12) => monthly
Sr>1 good, >2 better, >3 really good, <1 not so good

@param - returns(DataSeries): Daily returns
         rickFreeRate(float): The risk free rate of no investing, assume=0
         type(string): Daily Monthly Weekly or Yearly returns"""        
def sharpeRatio(returns, riskFreeRate=0, type=None):
    if type=='Daily':
        return np.sqrt(252)*(np.mean(returns-riskFreeRate))/np.std(returns)
    elif type=='Weekly':
        return np.sqrt(52)*(np.mean(returns-riskFreeRate))/np.std(returns)
    elif type=='Monthly':
        return np.sqrt(12)*(np.mean(returns-riskFreeRate))/np.std(returns)
    else:
        return (np.mean(returns)-riskFreeRate)/np.std(returns)

""" Create a portfolio
@params - stocks(array): An array of stocks
          start(Datetime): start datetime"""
def createPortfolio(stocks, start, end):
    dates = pd.date_range(start,end)
    df = pd.DataFrame(index=dates)
    invest = investment

    for s in stocks:
        adjClose = getHistoricCloses(s,start,end)
        df=df.join(adjClose,how='inner')
    return df

def plotPortfolioReturns(returns, title=None):
    #Figure size    
    returns.plot(figsize=(12,8))
    # Axis labels
    plt.xlabel('Year')
    plt.ylabel('Returns')
    # Create title
    if title is not None: plt.title(title)
    # Show plot
    plt.show()
    # Save plot in folder
    #plt.savefig('First plot', dpi=300)

start = datetime.datetime(2016,1,1)
end = datetime.datetime(2016,9,9)
dates = pd.date_range(start,end)
df = pd.DataFrame(index=dates)

initialInvestment = 1000000
stocks = ['MSFT','AMZN','FB','GOOGL']
weights = [0.2,0.4,0.2,0.2]

for s in stocks:
    adjClose = getHistoricCloses(s,start,end)
    df=df.join(adjClose,how='inner')
        
alloced = normalizingData(df)*weights
posValue = initialInvestment*alloced
portValue = posValue.sum(axis=1)
annualReturn = portValue[-1]/initialInvestment-1
portStd = np.std(portValue)
dr = dailyReturns(portValue)[1:]
sR = sharpeRatio(dr,0,type='Daily')
print('sharpeRatio:{}'.format(sR))
    
