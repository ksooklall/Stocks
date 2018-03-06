"""
RNN Network
"""
import tensorflow as tf
import numpy as np

class rnn_network():
    def __init__(self, serie, num_steps, input_size, embedding_size):
        """
        input_size: Amount of days to use in one prediction
        num_steps: The length of the unrolled network
        serie: Pandas serie of data
        """
        # embedding for multiple stocks
        self.serie = serie
        self.num_steps = num_steps
        self.input_size = input_size
        self.embedding_size = embedding_size

    def get_batch(self):
        # TODO: Add feature to over lap days
        start_index = len(self.serie) % self.input_size

        if not start_index:
            batch = self.serie.reshape(-1, self.input_size)
        else:
            # If a multiple of input_size is not len(serie) chop off from the start
            batch = self.serie[start_index:].reshape(-1, self.input_size)
        # TODO: Remove starting dates to get full batches
        X, y = [], []
        for i in range(0, len(batch) - self.num_steps):
            X.append(batch[i: i + self.num_steps])
            y.append(batch[i + self.num_steps])

        return (batch, np.array(X), np.array(y))

    def batch_generator(self, batch_X, batch_y, batch_size=32):
        for i in range(0, len(batch_X), batch_size):
            yield batch_X[i:i + batch_size], batch_y[i:i + batch_size]

    def create_batch(self, X, y):
        self.serie = X
        return self.get_batch(self)

    def get_train_test(self, split=0.1):
        _, X, y = self.get_batch()
        idx = int(np.round(len(X) * (1 - split)))
        X_train, X_test, y_train, y_test = X[:idx], X[idx:], y[:idx], y[idx:]
        return (X_train, X_test, y_train, y_test)

    def input_tensors(self, use_embeddings=True):
        if use_embeddings:
            inputs = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='inputs')
        else:
            inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.num_steps, self.input_size], name='inputs')
        targets = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size], name='targets')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return inputs, targets, learning_rate, keep_prob
        

    def build_lstm(self, lstm_size, layers, keep_prob):
        cell = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * layers)
        initial_state = cell.zero_state(32, dtype=tf.int32)
        return cell
    
    def build_output(self, lstm_size, output):
        last = tf.transpose(output, [1, 0 ,2])
        last = tf.gather(last, int(last.get_shape()[0])-1, name='lstm_state')
        with tf.variable_scope('output'):
            w = tf.Variable(tf.truncated_normal([lstm_size, self.input_size], stddev=0.01), name='w')
            b = tf.Variable(tf.truncated_normal([self.input_size], stddev=0.01), name='b')
            prediction = tf.matmul(last, w) + b
        return prediction

    def build_loss(self, logits, targets):
        loss = tf.reduce_mean(tf.square(tf.subtract(logits, targets)), name='loss')
        return loss

    def build_optimizer(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        t_vars = tf.trainable_variables()
        gradients, _ = tf.clip_by_global_norm(tf.gradients(loss, t_vars), 5)
        train_optimizer = optimizer.apply_gradients(zip(gradients, t_vars))
        return train_optimizer

    def set_save_times(self, counter):
        self.counter = counter
        return self.counter
        
        
    def train(self, lstm_size, layers, resume=False, batch_size=32, kp=0.2, epochs=10, lr=0.001, verbose=5):
        tf.reset_default_graph()
        inputs, targets, learning_rate, keep_prob = self.input_tensors()
        cell = self.build_lstm(lstm_size, layers, keep_prob)
        self.counter = self.counter if self.counter else epoch//10
        X_train, X_test, y_train, y_test = self.get_train_test()

        outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        prediction = self.build_output(lstm_size, outputs)
        loss = self.build_loss(prediction, targets)
        optimizer = self.build_optimizer(loss, learning_rate)
        # Add plotting and tensor board
        saver = tf.train.Saver(max_to_keep=10)
        with tf.Session() as sess:
            if resume:
                checkpoint = tf.train.latest_checkpoint('checkpoints')
                saver.restore(sess, checkpoint)
            else:
                sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                for X, y in self.batch_generator(X_train, y_train):
                    feed_dict = {inputs: X, targets: y, learning_rate: lr, keep_prob: kp}
                    l, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
                if verbose and not (epoch % verbose):
                    b_loss = sess.run([loss], feed_dict= feed_dict)
                    print('E: {:>3} b_loss: {:.3f}'.format(epoch, b_loss[-1]))

                if epoch and not (self.counter % epoch):
                    saver.save(sess, 'checkpoints/nile_{}{}{}{}.ckpt'.format(self.num_steps, self.input_size, lstm_size, epoch))
                    
            test_loss = sess.run([loss], feed_dict={inputs: X_test, targets: y_test, learning_rate:lr, keep_prob: kp})
            print('E: {:>3} test_loss: {:.3f}'.format(epoch, test_loss[-1]))
            saver.save(sess, 'checkpoints/nile_{}{}{}{}.ckpt'.format(self.num_steps, self.input_size, lstm_size, epoch))
        return sess
