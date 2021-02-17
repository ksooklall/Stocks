from datetime import datetime
import pandas as pd

date = str(datetime.now())[:10]
link = 'https://financhill.com/most-heavily-shorted-stocks-today'
df = pd.read_html(link)[0].dropna(how='all', axis=1)
df = df[df['Company'].notnull()]
df = df[~df['Company'].str.contains('Company')].drop(['Unnamed: 7'], axis=1)

df.to_csv('{}_shorted_stocks.csv'.format(date))
print(df)
