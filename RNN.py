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
        # TODO: Add feature to over lap days
        start_index = len(self.serie) % self.window

        if not start_index:
            batch = self.serie.reshape(-1, self.window)
        else:
            # If a multiple of window is not len(serie) chop off from the start
            batch = self.serie[start_index:].reshape(-1, self.window)
        # TODO: Remove starting dates to get full batches
        X, y = [], []
        for i in range(0, len(batch), self.num_steps):
            if (i + self.num_steps+ 1) > len(batch):
                break
            X.append(batch[i: i + self.num_steps])
            y.append(batch[i + self.num_steps])

        return (batch, np.array(X), np.array(y))

    def get_train_test(self, split=0.1):
        _, X, y = self.get_batch()
        idx = int(np.round(len(X) * (1 - split)))
        X_train, X_test, y_train, y_test = X[:idx], X[idx:], y[:idx], y[idx:]

        X_train = np.expand_dims(X_train, 0)
        
        return (X_train, X_test, y_train, y_test)

    def input_tensors(self):
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.window, self.num_steps])
        targets = tf.placeholder(dtype=tf.float32, shape=[None, self.window])
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        return inputs, targets, learning_rate
        
    
    def rnn_cell(self, units, layers):
        cell = tf.nn.rnn_cell.BasicLSTMCell(units)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * layers)
        initial_state = cell.zero_state(self.num_steps, dtype=tf.int32)
        return cell, initial_state
    
    def model(self, inputs, units=32, layers=1):
        cell, initial_state = self.rnn_cell(units, layers)
        import pdb; pdb.set_trace()
        output, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state)
        with tf.variable_scope('fc1'):
            fc1 = tf.contrib.full_connected(output, window, activation_fn=None)
        prediction = fc1
        return prediction, final_state

    def default_graph(self):
        with tf.Graph().as_default():
            inputs, targets, learning_rate = self.input_tensors()
            prediction, final_state = self.model(inputs)
            optimizer = tf.tain.AdamOptimizer(learning_rate)
            loss = tf.reduce_mean(tf.square(tf.subtract(prediction, targets)))
            gradients = optimizer.compute_gradients(loss)
            clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad, val in gradients]
            train_optimizer = optimizer.apply_gradients(clipped_gradients)
        return loss, train_optimizer

    def batch_generator(X, y, batch_size=32):
        for i in range(0, len(X), batch_size):
            yield X[i:i + batch_size], y[i:i + batch_size]

    def train(self, epochs, lr, verbose=5):
        X_train, X_test, y_train, y_test = self.get_train_test()
        loss, train_optimizer = self.default_graph()
        # TODO: Add batching
        with tf.Session() as sess:
            for epoch in epochs:
                for X, y in self.batch_generator(X_tain, y_train):
                    loss, _ = sess.run([loss, train_optimizer], feed_dict={inputs: X, targets: y, learning_rate: lr})
                if verbose and not (epoch % verbose):
                    acc = sess.run([loss], feed_dict={inputs: X_valid, targets: y_valid})
                    print('E:{:>3} Acc: {:.3f}'.format(epoch, acc))
        return    
