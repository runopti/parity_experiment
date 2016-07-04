import tensorflow as tf
import reader
import getData
import numpy as np
import math


np.random.seed(0)
tf.set_random_seed(0)

max_epoch = 100
hidden_size = 2 # 4
batch_size = 1
output_size = 2                                                                    
seq_len = 3  #6 

n = 1000


def calc_accuracy(n, state):
    total_sum = 0 
    for i in range(n):
        state = initial_state.eval() 
        x = getData.createInputData(seq_len)
        x = np.asarray(x).reshape(1, seq_len)
        y = getData.createTargetData(x[0])[-1]
        y_target = np.zeros((1,2))
        if y == 0: y_target[0][0] = 1
        else: y_target[0][1] = 1
        feed_dict={initial_state: state, data:x, target: y_target}
        output_ = session.run(output, feed_dict=feed_dict)
        #print(np.argmax(y_target))
        #print(np.argmax(output_))
        if np.argmax(y_target) == np.argmax(output_):
            total_sum += 1
    print("accuracy: %f" % (1.0 * total_sum / n))

input_data = getData.createInputData(n)   
target_data = getData.createTargetData(input_data)

############## Graph construction ################
data = tf.placeholder(tf.float32, [batch_size, seq_len])
target = tf.placeholder(tf.float32, [batch_size, output_size])

lstm = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
# initialize the state 
initial_state = state = tf.zeros([batch_size, hidden_size])

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
    with tf.name_scope("out") as scope:
        weights = tf.Variable(tf.truncated_normal([hidden_size, output_size], stddev=1.0/math.sqrt(float(hidden_size)), name="weights"))
        biases = tf.Variable(tf.truncated_normal([output_size], stddev=1.0/math.sqrt(float(hidden_size)), name="biases"))
        output = tf.nn.softmax(tf.matmul(cell_output, weights) + biases)  # the size of output should be just [batch_size, 1] right?
        #output = tf.reshape(output, [1,2])

loss = -tf.reduce_sum(target*tf.log(output)) # this should be just 1 by 1 - 1 by 1 
#train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

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
            #print(x[0])
            y = getData.createTargetData(x[0])[-1]
            y_target = np.zeros((1,2))
            #print(y_target)
            #print(y)
            if y == 0: y_target[0][0] = 1
            else: y_target[0][1] = 1
            #print(y_target)
            #exit()
            #numpy_state = initial_state.eval() 
            feed_dict={initial_state: numpy_state, data: x, target: y_target}
            numpy_state, current_loss, _, output_ = session.run([final_state, loss, train_op, output],
                    feed_dict=feed_dict)
            total_loss += current_loss
            
            #print(current_loss)
            #print("target: %d output: %d" % (np.argmax(y_target), np.argmax(output_[0])))
            #print(output_[0])
            #print(current_loss)
        print(total_loss)
        
        calc_accuracy(100, numpy_state)


        



