"""
RNN Network
"""
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np


class rnn_network():
    def __init__(self, serie, num_steps, input_size, embedding_size=None, stock_count=None):
        """
        serie: Pandas serie of data
        num_steps: The length of the unrolled rnn network
        input_size: Amount of days to use in one prediction
        embedding_size: Number of columns for embedding layer
        stock_count: Number of stocks to be analyez
        
        """
        # embedding for multiple stocks
        self.serie = serie
        self.num_steps = num_steps
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.stock_count = stock_count

    def get_batch(self, data):
        # TODO: Add feature to over lap days
        start_index = len(data) % self.input_size
        
        if not start_index:
            batch = data.reshape(-1, self.input_size)
        else:
            # If a multiple of input_size is not len(serie) chop off from the start
            batch = data[start_index:].reshape(-1, self.input_size)
        # TODO: Remove starting dates to get full batches
        
        X, y = [], []
        for i in range(0, len(batch) - self.num_steps):
            X.append(batch[i: i + self.num_steps])
            y.append(batch[i + self.num_steps])

        return (batch, np.array(X), np.array(y))

    def get_embedding_batch(self, batch_size=32):
        for stocks in range(1, self.stock_count+1):
            st = self.serie[self.serie[:, 1] == stocks]
            _, aX, ay = self.get_batch(st[:, 0])
            for i in range(0, len(aX), batch_size):
                yield aX[i:i+batch_size], ay[i:i+batch_size], stocks
    
    def batch_generator(self, batch_X, batch_y, batch_size=32):
        for i in range(0, len(batch_X), batch_size):
            yield batch_X[i:i + batch_size], batch_y[i:i + batch_size]

    def get_train_test(self, split=0.1):
        _, X, y = self.get_embedding_batch()
        idx = int(np.round(len(X) * (1 - split)))
        X_train, X_test, y_train, y_test = X[:idx], X[idx:], y[:idx], y[idx:]
        return (X_train, X_test, y_train, y_test)

    def input_tensors(self):
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.num_steps, self.input_size], name='inputs')
        targets = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size], name='targets')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        if self.embedding_size:
            ticker = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='ticker')
            return inputs, ticker, targets, learning_rate, keep_prob
        return inputs, targets, learning_rate, keep_prob
        

    def build_lstm(self, lstm_size, layers, keep_prob):
        with tf.variable_scope('rnn_cells'):
            cell = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, keep_prob)
            cell = tf.contrib.rnn.MultiRNNCell([cell] * layers)
        return cell

    def build_embedding_matrix(self, inputs, ticker):
        # Figure out how to reshape inputs to use embeddings
        with tf.variable_scope('embeddings'):
            self.embedding_matrix = tf.random_uniform([self.stock_count, self.embedding_size], minval=-1.0, maxval=1.0)
            tiled_embeddings = tf.tile(ticker, multiples=[1, self.num_steps], name='tiled')
            label_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, tiled_embeddings, name='emb_labels')

        inputs_with_embeddings = tf.concat([inputs, label_embeddings], axis=2)
        tf.summary.histogram('input_embed', inputs_with_embeddings)
        return inputs_with_embeddings
    
    def build_output(self, lstm_size, output):
        with tf.variable_scope('last_lstm'):
            last = tf.transpose(output, [1, 0 ,2])
            last = tf.gather(last, int(last.get_shape()[0])-1, name='lstm_state')
            tf.summary.histogram('last_lstm/lstm_state', last)
            
        with tf.variable_scope('output'):
            w = tf.Variable(tf.truncated_normal([lstm_size, self.input_size], stddev=0.1), name='w')
            b = tf.Variable(tf.truncated_normal([self.input_size], stddev=0.1), name='b')
            prediction = tf.matmul(last, w) + b
            tf.summary.histogram('w', w)
            tf.summary.histogram('b', b)
            tf.summary.histogram('add:0', prediction)

        return prediction

    def build_loss(self, logits, targets):
        loss = tf.reduce_mean(tf.square(tf.subtract(logits, targets)), name='loss')
        tf.summary.scalar('loss', loss)
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
       

    def merge_summaries(self, sess, path):
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(path, sess.graph)
        writer.add_graph(sess.graph)
        return merged, writer

    def debugging_vars(self):
        self.save = True
        return
    
    def train(self, lstm_size, layers, resume=False, batch_size=32, kp=0.8, epochs=10, lr=0.001, verbose=5):
        #tf.reset_default_graph()
        _count = 2
        if self.embedding_size:
            inputs, ticker, targets, learning_rate, keep_prob = self.input_tensors()
            embedding_inputs = self.build_embedding_matrix(inputs, ticker)
        else:
            inputs, targets, learning_rate, keep_prob = self.input_tensors()
            
        inputs = embedding_inputs if self.embedding_size else inputs
        cell = self.build_lstm(lstm_size, layers, keep_prob)
        self.counter = self.counter if self.counter else epoch//10
        #X_train, X_test, y_train, y_test = self.get_embedding_batch()
        self.debugging_vars()

        outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        prediction = self.build_output(lstm_size, outputs)
        loss = self.build_loss(prediction, targets)
        optimizer = self.build_optimizer(loss, learning_rate)

        # Add plotting and tensor board
        saver = tf.train.Saver(max_to_keep=10)
        with tf.Session() as sess:
            merged, writer = self.merge_summaries(sess, 'summaries/{}'.format(_count))

            # Set up embedding for tensorboard
            proj_config = projector.ProjectorConfig()
            proj_embed = proj_config.embeddings.add()
            proj_embed.tensor_name = self.embedding_matrix.name
            # Specify the width and height of a single thumbnail
            proj_embed.sprite.single_image_dim.extend([28, 28]) 
            projector.visualize_embeddings(writer, proj_config)

            if resume:
                checkpoint = tf.train.latest_checkpoint('checkpoints/{}'.format(_count))
                saver.restore(sess, checkpoint)
            else:
                sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                for X, y, stk in self.get_embedding_batch(batch_size=32):
                    feed_dict = {inputs: X, ticker: stk, targets: y, learning_rate: lr, keep_prob: kp}
                    import pdb; pdb.set_trace()
                    sess.run(optimizer, feed_dict=feed_dict)

                if verbose and not (epoch % verbose):
                    b_loss, s = sess.run([loss, merged], feed_dict= feed_dict)
                    writer.add_summary(s, epoch)
                    print('E: {:>3} b_loss: {:.3f}'.format(epoch, b_loss))

                if epoch and not (self.counter % epoch) and self.save:
                    saver.save(sess, 'checkpoints/nile_{}{}{}{}.ckpt'.format(self.num_steps, self.input_size, lstm_size, epoch))
                    
            test_loss = sess.run(loss, feed_dict={inputs: X_test, targets: y_test, learning_rate:lr, keep_prob: kp})
            print('E: {:>3} test_loss: {:.3f}'.format(epoch, test_loss))
            saver.save(sess, 'checkpoints/nile_{}{}{}{}.ckpt'.format(self.num_steps, self.input_size, lstm_size, epoch))
        return sess
