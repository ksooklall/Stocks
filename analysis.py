from datetime import datetime
import pandas as pd
import numpy as np
from RNN import rnn_network
import os

if __name__ == '__main__':
        master_df = pd.read_csv('E:/Analysis/sandp500/all_stocks_5yr.csv')
        df = master_df[master_df['Name'] == 'AMZN']
        serie = np.array(df['close'])
        net = rnn_network(serie, 3, 5)
        batch, X, y = net.get_batch()
        a = net.model()