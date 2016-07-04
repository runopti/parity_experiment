import tensorflow as tf
import reader
import getData
import numpy as np
import math

np.random.seed(0)
tf.set_random_seed(0)

max_epoch = 10
hidden_size = 3
batch_size = 1
output_size = 1
seq_len = 10
n = 10000
input_data = getData.createInputData(n)   
target_data = getData.createTargetData(input_data)

############## Graph construction ################
data = tf.placeholder(tf.float32, [batch_size, seq_len])
target = tf.placeholder(tf.float32, [batch_size])

lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
# initialize the state 
initial_state = state = tf.zeros([batch_size, 2*hidden_size])

for i in range(seq_len):
    if i > 0: tf.get_variable_scope().reuse_variables()
    with tf.name_scope("in") as scope:
        weights = tf.Variable(tf.truncated_normal([hidden_size, batch_size], stddev=1.0/math.sqrt(float(hidden_size)), name="weights"))
        biases = tf.Variable(tf.truncated_normal([hidden_size], stddev=1.0/math.sqrt(float(hidden_size)), name="biases"))
        mapped = weights *  data[:,i]
        mapped = tf.reshape(mapped, [1, hidden_size]) # 1 x hidden_size
        mapped = mapped + biases
        #mapped = tf.Print(mapped, [mapped], message="this is mapped: ")
    cell_output, state = lstm(mapped, state) 
    with tf.name_scope("in") as scope:
        weights = tf.Variable(tf.truncated_normal([hidden_size, output_size], stddev=1.0/math.sqrt(float(hidden_size)), name="weights"))
        biases = tf.Variable(tf.truncated_normal([output_size], stddev=1.0/math.sqrt(float(hidden_size)), name="biases"))
        output = tf.matmul(cell_output, weights) + biases  # the size of output should be just [batch_size, 1] right?

loss = tf.reduce_mean(tf.square(output - target)) # this should be just 1 by 1 - 1 by 1 
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

final_state = state

init_op = tf.initialize_all_variables()
############### Graph construction end ##########

with tf.Session() as session:
    session.run(init_op)
    numpy_state = initial_state.eval() 
    for epoch in range(max_epoch):
        total_loss = 0
        for step, (x,y) in enumerate(reader.parity_iterator(input_data,target_data,batch_size, seq_len)):
            #print([y[0][-1]])
            #exit()
            feed_dict={initial_state: numpy_state, data: x, target: [y[0][-1]]}
            numpy_state, current_loss, _ = session.run([final_state, loss, train_op],
                    feed_dict=feed_dict)
            total_loss += current_loss
            #print(current_loss)
        print(total_loss)
