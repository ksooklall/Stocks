import pandas as pd

def convert_to_float(df, columns, to_replace=r'\$|,', value=''):
    """
    Convert a series $453,534.65 --> 453.543.65 with sep=','
    """
    for col in columns:
        df[col] = df[col].replace(to_replace, value=value, regex=True)
    df[columns] = df[columns].apply(pd.to_numeric)
    return df

def clean_columns(df):
    """
    Lowers the case of all columns and removes spaces
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df
    
def preprocess_df(df, float_cols, **kwrgs):
	df = clean_columns(df)
	df = convert_to_float(df, float_cols, **kwrgs)
	return df
