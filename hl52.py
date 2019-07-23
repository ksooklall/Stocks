import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
from PandasUtility import preprocess_df
from yahoofinancials import YahooFinancials

def get_financial_data(tickers):
	financial_columns = ['date', 'symbol', 'netTangibleAssets', 'extraordinaryItems', 'ebit', 'commonStock', 'totalLiab', 'incomeBeforeTax', 'otherCurrentAssets', 'capitalExpenditures', 'netIncomeFromContinuingOps', 'issuanceOfStock', 'otherCashflowsFromInvestingActivities', 'otherOperatingExpenses', 'treasuryStock', 'researchDevelopment', 'effectOfAccountingCharges', 'changeToOperatingActivities', 'depreciation', 'dividendsPaid', 'capitalSurplus', 'changeToInventory', 'changeInCash', 'totalStockholderEquity', 'discontinuedOperations', 'otherStockholderEquity', 'totalCashflowsFromInvestingActivities', 'propertyPlantEquipment', 'inventory', 'netBorrowings', 'costOfRevenue', 'repurchaseOfStock', 'totalRevenue', 'totalOperatingExpenses', 'otherItems', 'minorityInterest', 'otherAssets', 'totalCurrentLiabilities', 'grossProfit', 'sellingGeneralAdministrative', 'incomeTaxExpense', 'investments', 'changeToAccountReceivables', 'effectOfExchangeRate', 'otherCurrentLiab', 'operatingIncome', 'shortTermInvestments', 'otherCashflowsFromFinancingActivities', 'netIncomeApplicableToCommonShares', 'cash', 'longTermInvestments', 'changeToNetincome', 'totalOtherIncomeExpenseNet', 'retainedEarnings', 'otherLiab', 'totalCashFromOperatingActivities', 'longTermDebt', 'netReceivables', 'nonRecurring', 'totalCurrentAssets', 'totalCashFromFinancingActivities', 'accountsPayable', 'shortLongTermDebt', 'interestExpense', 'netIncome', 'totalAssets', 'goodWill', 'intangibleAssets', 'changeToLiabilities']
       
	print('Initiating yahoo financial api')
	fin_dict = YahooFinancials(tickers)
	print('Getting quaterly statements')
	import pdb; pdb.set_trace()
	stmt_data = fin_dict.get_financial_stmts('quaterly', ['cash', 'balance', 'income'])
	df = pd.DataFrame()
	for ticker in tickers:
		print('Parsing {} financial statements'.format(ticker))
		income_df = pd.DataFrame(stmt_data['incomeStatementHistoryQuarterly'][ticker][0]).fillna(-1).T
		balance_df = pd.DataFrame(stmt_data['balanceSheetHistoryQuarterly'][ticker][0]).fillna(-1).T
		cash_df = pd.DataFrame(stmt_data['cashflowStatementHistoryQuarterly'][ticker][0]).fillna(-1).T
		tdf = pd.concat([income_df, balance_df, cash_df], axis=1).reset_index().rename(columns={'index': 'date'})
		tdf['symbol'] = ticker
		if tdf.shape[-1] != len(financial_columns):
			missing_cols = dict.fromkeys(list(set(financial_columns) - set(tdf.columns)), -1)
			import pdb; pdb.set_trace()		
			tdf = tdf.assign(**missing_cols)
			# Remove duplicated columns
		tdf = tdf.loc[:, ~tdf.columns.duplicated()]
		import pdb; pdb.set_trace()		
		df = pd.concat([df, tdf], axis=0)

def get_statistical_data(tickers):
	yahoo_obj = YahooFinancials(tickers)
	print('Getting statistical information')
	stat_data = yahoo_obj.get_key_statistics_data()
	df = pd.concat([pd.DataFrame(v, index=[k]) for i, (k, v) in enumerate(stat_data.items())])
	return df


def value_investing(tickers):
	"""
	1 - Price to earnings (pe ratio) - What you pay for $1 of earning, compare to sector average
	2 - Price to book ratio (pb ratio) - Current share price / equity per share (book value)
												below 1 -> less than value of assets
												above 1 -> more than value of assets
	3 - Price to earning to growth (PEG):
			Below 1 -> Undervalued
	4 - Return on Equity (ROE) 
				17-20% 	very good 
				20-25% 	excellent
				30%+ 	superior
	5 - Debt to equity ratio
				Total debt / shareholder equity
				The lower to better, avoid above 2
	6 - Current ratio 'liquidity ratio' - 	below 1 -> liabilities exceed assets
											above 1 -> assets exceed liabilities
											above 3 -> holding back
											best to be between 1 - 2
	"""
	rename_cols = {	'Forward P/E 1': 'forwardPE', 
					'Price/Book (mrq)': 'pbRatio', 
					'PEG Ratio (5 yr expected) 1': 'pegRatio', 
					'Return on Equity (ttm)': 'returnOnEquity', 
					'Total Debt/Equity (mrq)': 'debtToEquity',
					'Current Ratio (mrq)': 'currentRatio'}
	#df = get_statistical_data(tickers)

	def get_statistics(ticker):
		url = 'https://finance.yahoo.com/quote/{0}/key-statistics?p={0}'.format(ticker)
		ldf = pd.read_html(url)
		df = pd.concat([i for i in ldf]).set_index([0], drop=True).rename(columns={1: ticker})
		return df
	df = pd.DataFrame()

	for ticker in tickers:
		print('Getting {} statistics'.format(ticker))
		tdf = get_statistics(ticker)
		df = pd.concat([df, tdf], axis=1)
	import pdb; pdb.set_trace()

print('Getting 52 week low stocks ....')
ldf = pd.read_html('https://www.nasdaq.com/aspx/52-week-high-low.aspx?exchange=NASDAQ&status=LOW')[0]
ldf = ldf[ldf['Symbol'].str.len() < 5]
print('Getting 52 week high stocks ....')
hdf = pd.read_html('https://www.nasdaq.com/aspx/52-week-high-low.aspx?exchange=NASDAQ&status=HIGH')[0]
hdf = hdf[hdf['Symbol'].str.len() < 5]

ldf = preprocess_df(ldf, float_cols=['new_low', 'previous_low', 'high'])
hdf = preprocess_df(hdf, float_cols=['new_high', 'previous_high', 'previous_low'])


symbols = ldf['symbol'] + hdf['symbol']
#get_financial_data(['CALM', 'MSFT'])
value_investing(['GOOGL', 'ARLO', 'MAXR', 'AMZN', 'TSLA'])
import pdb; pdb.set_trace()

