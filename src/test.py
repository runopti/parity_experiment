import tensorflow as tf
import reader
import getData
import numpy as np
import math
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# for storing info
import platform 
import sys
import datetime
import os

info = {}
info['script_name'] = sys.argv[0].split('/')[-1]
info['python_version'] = platform.python_version()
info['sys_uname'] = platform.uname()

start_time = datetime.datetime.now()
start_utime = os.times()[0]
info['start_time'] = start_time.isoformat()

import argparse                                                                          
parser = argparse.ArgumentParser(description='Define parameters for Parity Experiment.')
parser.add_argument('--max_epoch', type=int)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--output_size', type=int)
parser.add_argument('--seq_len', type=int)
parser.add_argument('--data_size', type=int)
parser.add_argument('--rseed', type=int)

args = parser.parse_args()

for i in args.__dict__.keys():
    if i[0] != "_":
        info['option_' + i] = str(getattr(args, i))

max_epoch = args.max_epoch  # 300
hidden_size = args.hidden_size # 2 # 4
batch_size = args.batch_size #1
output_size = args.output_size  #2
seq_len = args.seq_len  #12 #3  #6 

n = args.data_size # 1000
print(max_epoch)
np.random.seed(args.rseed)
tf.set_random_seed(args.rseed)


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
    return (1.0 * total_sum / n)
    #print("accuracy: %f" % (1.0 * total_sum / n))

input_data = getData.createInputData(n)
target_data = getData.createTargetData(input_data)

with tf.Graph().as_default():
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
    tf.scalar_summary("loss", loss)
    #train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

    tf.add_to_collection('train_op', train_op)
    tf.add_to_collection('output', output)
    tf.add_to_collection('initial_state', initial_state)
    tf.add_to_collection('data', data)
    tf.add_to_collection('target', target)

    final_state = state

    summary_op = tf.merge_all_summaries()
    init_op = tf.initialize_all_variables()
    ############### Graph construction end ##########

    total_loss_list = []
    acc_list = []
    saver = tf.train.Saver()
    with tf.Session() as session:
        summary_writer = tf.train.SummaryWriter("tensorflow_log", graph=session.graph)
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
                #if step % 100 == 0: 
                    #print("loss")
                    #current_loss = session.run(loss, feed_dict)
                #    summary_str = session.run(summary_op, feed_dict)
                #    summary_writer.add_summary(summary_str, step)

                    #print(current_loss)
            print(total_loss)
            total_loss_list.append(total_loss)

            acc = calc_accuracy(100, numpy_state)
            acc_list.append(acc)
        saver.save(session, 'my_model', global_step=0)


import pickle
with open('total_loss_list.pickle', 'wb') as f:
    pickle.dump(total_loss_list, f)

with open('acc_list.pickle', 'wb') as f:
    pickle.dump(acc_list, f)

end_time = datetime.datetime.now()
end_utime = os.times()[0]                                                                                                                                                          
info['end_time'] = end_time.isoformat()
info['elapsed_time'] = str((end_time - start_time))
info['elapsed_utime'] = str((end_utime - start_utime))

with open('info.txt', 'w') as outfile:
    for key in info.keys():
        outfile.write("#%s=%s\n" % (key, str(info[key])))
