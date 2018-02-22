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

if __name__ == '__main__':
        EPOCH = 10
        LEARNING_RATE = 0.001
        
        master_df = pd.read_csv('E:/Analysis/sandp500/all_stocks_5yr.csv')
        df = master_df[master_df['Name'] == 'AMZN']
        serie = np.array(df['close'])
        
        net = rnn_network(serie, 2, 5)
        seq, lx, ly = lil(serie)
        batch, X, y = net.get_batch()
        net.train(EPOCH, LEARNING_RATE)

        
        
