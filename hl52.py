import pandas as pd
from PandasUtility import preprocess_df
from yahoofinancials import YahooFinancials

def get_financial_data(tickers):
	financial_columns = ['date', 'symbol', 'netTangibleAssets', 'extraordinaryItems', 'ebit', 'commonStock', 'totalLiab', 'incomeBeforeTax', 'otherCurrentAssets', 'capitalExpenditures', 'netIncomeFromContinuingOps', 'issuanceOfStock', 'otherCashflowsFromInvestingActivities', 'otherOperatingExpenses', 'treasuryStock', 'researchDevelopment', 'effectOfAccountingCharges', 'changeToOperatingActivities', 'depreciation', 'dividendsPaid', 'capitalSurplus', 'changeToInventory', 'changeInCash', 'totalStockholderEquity', 'discontinuedOperations', 'otherStockholderEquity', 'totalCashflowsFromInvestingActivities', 'propertyPlantEquipment', 'inventory', 'netBorrowings', 'costOfRevenue', 'repurchaseOfStock', 'totalRevenue', 'totalOperatingExpenses', 'otherItems', 'minorityInterest', 'otherAssets', 'totalCurrentLiabilities', 'grossProfit', 'sellingGeneralAdministrative', 'incomeTaxExpense', 'investments', 'changeToAccountReceivables', 'effectOfExchangeRate', 'otherCurrentLiab', 'operatingIncome', 'shortTermInvestments', 'otherCashflowsFromFinancingActivities', 'netIncomeApplicableToCommonShares', 'cash', 'longTermInvestments', 'changeToNetincome', 'totalOtherIncomeExpenseNet', 'retainedEarnings', 'otherLiab', 'totalCashFromOperatingActivities', 'longTermDebt', 'netReceivables', 'nonRecurring', 'totalCurrentAssets', 'totalCashFromFinancingActivities', 'accountsPayable', 'shortLongTermDebt', 'interestExpense', 'netIncome', 'totalAssets', 'goodWill', 'intangibleAssets', 'changeToLiabilities']
       
	print('Initiating yahoo financial api')
	fin_dict = YahooFinancials(tickers)
	print('Getting quaterly statements')
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


print('Getting 52 week low stocks ....')
ldf = pd.read_html('https://www.nasdaq.com/aspx/52-week-high-low.aspx?exchange=NASDAQ&status=LOW')[0]
ldf = ldf[ldf['Symbol'].str.len() < 5]
print('Getting 52 week high stocks ....')
hdf = pd.read_html('https://www.nasdaq.com/aspx/52-week-high-low.aspx?exchange=NASDAQ&status=HIGH')[0]
hdf = hdf[hdf['Symbol'].str.len() < 5]

ldf = preprocess_df(ldf, float_cols=['new_low', 'previous_low', 'high'])
hdf = preprocess_df(hdf, float_cols=['new_high', 'previous_high', 'previous_low'])


symbols = ldf['symbol'] + hdf['symbol']
get_financial_data(['CALM', 'MSFT'])
import pdb; pdb.set_trace()

