from datetime import datetime
import pandas as pd
import numpy as np
from RNN import rnn_network
import os

def lil(serie, num_steps=2, input_size=5):
        input_size = 5
        seq = [np.array(serie[i * input_size: (i + 1) * input_size]) for i in range(len(serie) // input_size)]
        X = np.array([seq[i: i + num_steps] for i in range(len(seq) - num_steps)])
        y = np.array([seq[i + num_steps] for i in range(len(seq) - num_steps)])
        return seq, X, y

def normalize(X):
        _max, _min = X.max(), X.min()
        return (X - X.min())/ (X.max() - X.min()), _max, _min

if __name__ == '__main__':
        EPOCH = 100
        LEARNING_RATE = 0.001
        
        master_df = pd.read_csv('E:/Analysis/sandp500/all_stocks_5yr.csv')
        df = master_df[master_df['Name'] == 'AMZN']
        serie = np.array(df['close'])
        serie, _max, _min = normalize(serie)
        net = rnn_network(serie, 5, 30)
        seq, lx, ly = lil(serie)
        batch, X, y = net.get_batch()
        net.train(32, 1, 0.2, EPOCH, LEARNING_RATE)

        
        
