import tensorflow as tf
import os
import shutil
from datetime import datetime, timedelta

from .train_log import *


class Config():
    def __init__(self):
        pass

class LSTMTextGenerator(tf.Graph):
    def __init__(self, cnfg):
        super().__init__()
        self.build(cnfg)

    def apply_batch_normalize(self, inputs):
        return tf.contrib.layers.batch_norm(inputs=inputs,
                                            updates_collections=None,
                                            is_training=self.is_training,
                                            scope='bn')

    def get_weight_tensor(self, shape, stddev):
        return tf.get_variable(name='W', shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    
    def get_bias_tensor(self, shape, value):
        return tf.get_variable(name='b', shape=shape, initializer=tf.constant_initializer(value=value))
        
    def apply_bn_dr_XWplusb(self, X, W_shape, W_stddev=0.015, b_value=0.1, skip_relu=False):
        bn_ = self.apply_batch_normalize(X)
        dropout_ = tf.nn.dropout(bn_, keep_prob=self.keep_prob)
        W = self.get_weight_tensor(W_shape, W_stddev)
        b = self.get_bias_tensor(W_shape[1], b_value)
        XWplusb = tf.matmul(dropout_, W) + b
        
        if skip_relu:
            return XWplusb
        else:
            return tf.nn.relu(XWplusb)       
        
    def build(self, cnfg):
        self.cnfg = cnfg
        
        with self.as_default():
            global_step = tf.Variable(0, trainable=False)
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)
            
            self.X = tf.placeholder(tf.float32, [None, cnfg.n_char_per_memory, cnfg.n_char])
            self.Y = tf.placeholder(tf.float32, [None, cnfg.n_char])
            
            batch_size = tf.shape(self.X)[0]

            with tf.variable_scope('lstm'):
                X_lstm = self.apply_batch_normalize(self.X)
                # (batch size) x (length of time) x (dim of data at each time)
                
                cell_list = []
                initial_state_list = []
                input_sizes = [cnfg.n_char] + cnfg.lstm_state_sizes
                for i in range(len(cnfg.lstm_state_sizes)):
                    cell = tf.contrib.rnn.BasicLSTMCell(cnfg.lstm_state_sizes[i])  # forget_bias=1.0
                    cell = tf.contrib.rnn.DropoutWrapper(cell, input_size=input_sizes[i], dtype=tf.float32,
                                                         input_keep_prob=self.keep_prob,
                                                         output_keep_prob=1.0,
                                                         state_keep_prob=1.0,
                                                         variational_recurrent=True)                 
                    initial_state = cell.zero_state(batch_size, tf.float32)
                    cell_list.append(cell)
                    initial_state_list.append(initial_state)
                    
                multi_cell = tf.contrib.rnn.MultiRNNCell(cells=cell_list)
                outputs, final_state = tf.nn.dynamic_rnn(multi_cell, X_lstm, initial_state=tuple(initial_state_list))
                # outputs is (batch size) x (length of time) x (lstm state size)
                final_output = outputs[:, -1, :]

            n_flat = cnfg.lstm_state_sizes[-1]
            X1 = tf.reshape(final_output, [-1, n_flat])
                
            with tf.variable_scope('flat'):
                self.logits = self.apply_bn_dr_XWplusb(X=X1, W_shape=[n_flat, cnfg.n_char], skip_relu=True)
                self.softmax = tf.nn.softmax(self.logits)  # Not for train, just for generate
                
            # Accuracy
            correct_or_not = tf.cast(tf.equal(tf.argmax(self.Y, axis=1), tf.argmax(self.logits, axis=1)), tf.float32)
            # 1 if correct, 0 if not
            
            self.accuracy = tf.reduce_mean(correct_or_not)

            # Logloss
            L = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits)
            self.logloss = tf.reduce_mean(L)
            
            # Optimization
            self.optimizer = tf.train.GradientDescentOptimizer(cnfg.lr).minimize(self.logloss, global_step=global_step)
            
    def train_model(self, cnfg, train, test):  # train and test are instances of Text class        
        log = Log(cnfg)
        
        if os.path.isdir(cnfg.ckp_dir):
            shutil.rmtree(cnfg.ckp_dir)
        os.makedirs(cnfg.ckp_dir)
        
        with tf.Session(graph=self) as self.sess: 
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()

            for step in range(1, cnfg.max_step):
                if step == 1:  # Only first time
                    print('Training starts @ {:%m/%d/%Y %H:%M:%S}'.format(datetime.now()))
                    print('=' * 100)
                    
                    self.last_saved_time = datetime.now()
                    
                X_batch, Y_batch = train.get_next_batch()                
                _, ll, accu = self.sess.run([self.optimizer, self.logloss, self.accuracy], 
                                            feed_dict={self.X: X_batch, self.Y: Y_batch, 
                                                       self.keep_prob: cnfg.dropout_keep_prob, 
                                                       self.is_training: True})
                log.record(step, ll, accu)
                
                if (step == 1) or (step % cnfg.generate_every == 0):  # See what current model generates
                    self.generate(test, cnfg.n_generate)
                    print('Step {:,} ends @ {:%m/%d/%Y %H:%M:%S} [Logloss] {:.3f} [Accuracy] {:.1f}%'.format(step, datetime.now(), ll, accu*100))
                    print('-' * 100)
                    print(test)
                    print('=' * 100)
                    test.reset()
                    
                if datetime.now() > self.last_saved_time + timedelta(seconds=10*60):  # Save model every 10 min
                    saver.save(self.sess, '/'.join([cnfg.ckp_dir, 'model']), global_step=step)
                    self.last_saved_time = datetime.now()

            saver.save(self.sess, '/'.join([cnfg.ckp_dir, 'model']), global_step=step)
            print('Training ends @ {:%m/%d/%Y %H:%M:%S}'.format(datetime.now()))  # The end

    def generate(self, test, n):
        test.set_attr(chars=self.cnfg.chars, char2id=self.cnfg.char2id, 
                      id2char=self.cnfg.id2char, n_char_per_memory=self.cnfg.n_char_per_memory)
        for _ in range(n):
            input_ = test.get_input_for_generate()
            softmax = self.sess.run(self.softmax, feed_dict={self.X: input_, self.keep_prob: 1.0, self.is_training: False})[0]
            test.add_char_from_softmax(softmax)

    def load_and_generate(self, ckp_dir, test, n):
        path2ckp = tf.train.latest_checkpoint(ckp_dir)
        
        with tf.Session(graph=self) as self.sess:
            tf.global_variables_initializer().run()
            
            saver = tf.train.Saver()
            saver.restore(self.sess, path2ckp)  # Load model
            
            self.generate(test, n)
            