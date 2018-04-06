from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from RNN import rnn_network
import os
import matplotlib.pyplot as plt

def lil(serie, num_steps=2, input_size=5):
        input_size = 5
        seq = [np.array(serie[i * input_size: (i + 1) * input_size]) for i in range(len(serie) // input_size)]
        X = np.array([seq[i: i + num_steps] for i in range(len(seq) - num_steps)])
        y = np.array([seq[i + num_steps] for i in range(len(seq) - num_steps)])
        return seq, X, y

def normalize(X):
        _max, _min = X.max(), X.min()
        return (X - X.min())/ (X.max() - X.min()), _max, _min

def predict(graph_name, test_X, test_y):
    """
    Restore a pretrained model
    Compute residual = model output - actual value
    and plot histogram
    Plot model predictions for both training and testing set
    https://github.com/ksooklall/Artificial-Intelligence/blob/master/Term_2/aind2-rnn/RNN_project.ipynb
    line 32
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('checkpoints/{}'.format(graph_name))
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

        graph = tf.get_default_graph()
        test_feed_dict = {graph.get_tensor_by_name('inputs:0'): test_X,
                          graph.get_tensor_by_name('targets:0'): test_y,
                          graph.get_tensor_by_name('learning_rate:0'): 0.001,
                          graph.get_tensor_by_name('keep_prob:0'): 0.2}
        pred = graph.get_tensor_by_name('output/add:0')
        loss = graph.get_tensor_by_name('loss:0')
        test_pred, test_loss = sess.run([pred, loss], test_feed_dict)
    return test_pred, test_loss

def plotting(true, pred, bins, kind='hist'):
    """
    Plotting
    The residual(res) is show the model (nile_251699) is predicting values above the true
    """
    if kind == 'hist':
      res = true - pred
      plt.title('AMZN norm')
      plt.xlabel('true-pred')
      plt.ylabel('Count {} bins'.format(bins))
      plt.hist(res, bins)
    elif kind =='scatter':
      plt.scatter(pred, y, s=20)
    else:
      fig, ax = plt.subplots(figsize=(15,5))
      plt.plot(y, color='k')
      plt.plot(pred, color='b', alpha=0.5)
      plt.xlabel('Normalized price')
      plt.ylabel('Days')
    plt.show()
    return true - pred

def denorm(a, ma, mi):
    return a * (ma-mi) + mi
        
if __name__ == '__main__':
    epochs = 100
    learning_rate = 0.001
    lstm_size = 32
    layers = 1
    keep_prob = 0.8
    num_steps = 2
    input_size = 5
    embedding_size = 100
    stock_count = 20
    
    master_df = pd.read_csv('E:/Analysis/sandp500/all_stocks_5yr.csv')
    master_df['ticker_labels'] = master_df['Name'].astype('category').cat.codes
    tickers = master_df['Name'].unique()

    df = master_df[master_df['Name'].isin(tickers[:stock_count])]   
    serie = np.array(df['close'])
    serie, _max, _min = normalize(serie)
    serie = np.append(serie.reshape(-1, 1), np.array(df['ticker_labels']).reshape(-1, 1), axis=1)
    
    net = rnn_network(serie, num_steps, input_size, embedding_size=embedding_size, stock_count=stock_count)
    seq, X, y = lil(serie)
    net.set_save_times(50)

    net.train(lstm_size=lstm_size, layers=layers, resume=False, batch_size=32, kp=keep_prob, epochs=epochs, lr=learning_rate, verbose=20)
    #pred, loss = predict('nile_251699.ckpt.meta', X, y)
    #up, uy = denorm(pred, _max, _min), denorm(y, _max, _min)
    #res = plotting(uy, up, bins=10, kind='hist')
