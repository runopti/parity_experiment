import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import math

class RNNmodel:
    def __init__(self, seq_len, target_size, hidden_size, batch_size):
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.target_size = target_size 
        self.batch_size = batch_size 
        #self.lstm_size = lstm_size -> this is a hidden size I guess?
        print(self.hidden_size)

        self.input = tf.placeholder(tf.float32, shape=(batch_size, seq_len))
        self.target = tf.placeholder(tf.float32, shape=(batch_size, seq_len))

        self.state = tf.zeros([self.batch_size, 2*self.hidden_size])
        self.output, self.hidden_state = self._inference(self.input, self.state)
        self.loss = loss = self._loss(self.output, self.target)

        self.train = self._training(loss)

        self.init_op = tf.initialize_all_variables()
        self.summary_op = tf.merge_all_summaries()
        
    def _inference(self, input, hidden_state):
        with tf.name_scope("hidden_in") as scope:
            weights = tf.Variable(tf.truncated_normal([self.seq_len, self.hidden_size], stddev=1.0/math.sqrt(float(self.seq_len)), name="weights"))
            biases = tf.Variable(tf.zeros([self.hidden_size], name="biases"))
            hidden_in = tf.matmul(input, weights) + biases
        with tf.name_scope("LSTM") as scope:
            lstm = rnn_cell.BasicLSTMCell(self.hidden_size)
            cell_output, hidden_state = lstm(hidden_in, hidden_state)
        with tf.name_scope("hidden_out") as scope:
            weights = tf.Variable(tf.truncated_normal([self.hidden_size, self.target_size], stddev=1.0/math.sqrt(float(self.target_size)), name="weights"))
            biases = tf.Variable(tf.zeros([self.target_size], name="biases"))
            output = tf.matmul(cell_output, weights) + biases
            return output, hidden_state

    def _loss(self, output, target):
        with tf.name_scope("mse-loss") as scope:
            loss = tf.reduce_mean(tf.square(output - target))
            tf.scalar_summary("mse-loss", loss)
            return loss

    def _training(self, loss):
        with tf.name_scope("training") as scope:
            train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
            return train_step


    @property
    def output_op(self):
        return self.output

    @property
    def state_op(self):
        return self.hidden_state
            
    @property
    def loss_op(self):
        return self.loss

    @property
    def train_op(self):
        return self.train

    @property
    def get_summary_op(self):
        return self.summary_op
        
    @property
    def get_init_op(self):
        return self.init_op

