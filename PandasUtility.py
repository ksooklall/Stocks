import pandas as pd

def convert_float(df, columns, to_replace=r'\$|,', value=''):
    """
    Convert a series $453,534.65 --> 453.543.65 with sep=','
	@params df (DataFrame): DataFrame to be modified
	@params columns (list): List of columns
    """
    for col in columns:
        df[col] = df[col].replace(to_replace, value=value, regex=True)
    df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')
    return df

def clean_columns(df):
    """
    Lowers the case of all columns and removes spaces
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def convert_kmb_float(df, columns, mapping_dict={'K': 1e3, 'M': 1e6, 'B': 1e9}):
	# Convert columns containg K - Thousand, M - Million or B - Billion at the end to float
	for col in columns:
		df[col] = df[col].str.replace(' ','').str.upper().str.replace('K|B|M', '').astype(float).mul(df[col].str.upper().str.get(-1).map(mapping_dict))
	return df

 
def preprocess_df(df, float_cols, **kwrgs):
	df = clean_columns(df)
	df = convert_float(df, float_cols, **kwrgs)
	return df
