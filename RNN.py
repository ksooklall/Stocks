"""
RNN Network
"""
import tensorflow as tf
import numpy as np

class rnn_network():
    def __init__(self, serie, num_steps, window):
        self.serie = serie
        self.num_steps = num_steps
        self.window = window

    def get_batch(self):
        # TO DO: Add feature to over lap days
        start_index = len(self.serie) % self.window

        if not start_index:
            batch = self.serie.reshape(-1, self.window)
        else:
            # If a multiple of window is not len(serie) chop off from the start
            batch = self.serie[start_index:].reshape(-1, self.window)
        X, y = [], []
        for i in range(0, len(batch), self.num_steps):
            if (i + self.num_steps+ 1) > len(batch):
                break
            X.append(batch[i: i + self.num_steps])
            y.append(batch[i + self.num_steps])

        return batch, np.array(X), np.array(y)

    def tensors(self):
        inputs = tf.placeholder([None, self.num_steps, self.window], dtype=tf.float32)
        targets = tf.placeholder([None, self.window], dtype=tf.float32)
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        return inputs, targets
        
    
    def rnn_cell(self, units, layers):
        cell = tf.contrib.rnn.BasicLSTMCell(units)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * layers)
        initial_state = cell.zero_state(self.num_steps, dtype=tf.int32)
        return cell, initial_state
    
    def model(self, units=32, layers=1):
        cell, initial_state = self.rnn_cell(units, layers)
        output, final_state = tf.nn.dynamic_rnn(cell, X, initial_state)
        with tf.variable_scope('fc1'):
            fc1 = tf.contrib.full_connected(output, window, activation_fn=None)
        return fc1

    def train(self):
        inputs, targets = self.tensors()
        pass
