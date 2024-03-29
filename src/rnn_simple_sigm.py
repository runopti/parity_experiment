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
parser.add_argument('--lr_test', type=bool, default=False)
parser.add_argument('--loss_diff_eps', type=float)
parser.add_argument('--grad_clip', type=bool)
parser.add_argument('--max_grad_norm', type=float)
parser.add_argument('--train_method', type=str)
parser.add_argument('--cell_type', type=str, default='basic')
parser.add_argument('--learning_rate', type=float, default=0.001)


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
        feed_dict={initial_state: state, data:x}
        output_ = session.run(output, feed_dict=feed_dict)
        #print(np.argmax(y_target))
        #print(np.argmax(output_))
        if np.argmax(y_target) == np.argmax(output_):
            total_sum += 1
    return (1.0 * total_sum / n)
    #print("accuracy: %f" % (1.0 * total_sum / n))

input_data = getData.createInputData(n)
target_data = getData.createTargetData(input_data)

val_input_data = getData.createInputData(n)
val_target_data = getData.createTargetData(val_input_data)

with tf.Graph().as_default():
    ############## Graph construction ################
    data = tf.placeholder(tf.float32, shape=[batch_size, seq_len])
    if args.train_method == "single":
        target = tf.placeholder(tf.float32, shape=[batch_size, output_size])
    else:
        target = tf.placeholder(tf.float32, shape=[seq_len, batch_size, output_size])
        target_list = [tf.squeeze(tf.slice(target, [i,0,0], [1, batch_size, output_size])) for i in range(seq_len)]
    learning_rate = tf.placeholder(tf.float32, shape=[])

    # define cell and initialize the cell state
    if args.cell_type == 'basic':
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        initial_state = state = tf.zeros([batch_size, hidden_size])
    elif args.cell_type == 'lstm':
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        initial_state = state = tf.zeros([batch_size, 2*hidden_size])
    else:
        print("Error: Define cell_type")
    
    if args.train_method != "single":
        loss = tf.constant(0.0)
    for i in range(seq_len):
        if i > 0: tf.get_variable_scope().reuse_variables()
        cell_output, state = rnn_cell(tf.reshape(data[:,i], [1,1]), state) # should change [1,1] if I want to change batch_size?
        # I need to use tf.get_variable to activate reuse_variables()
        weights = tf.get_variable("weights", dtype=tf.float32, initializer=tf.truncated_normal([hidden_size, output_size], stddev=1.0/math.sqrt(float(hidden_size))))
        biases = tf.get_variable("biases", dtype=tf.float32,initializer=tf.truncated_normal([output_size], stddev=1.0/math.sqrt(float(hidden_size))))
        #output = tf.nn.softmax(tf.matmul(cell_output, weights) + biases) # the size of output should be just [batch_size, 1] right?
        output = tf.nn.sigmoid(tf.matmul(cell_output, weights) + biases) # the size of output should be just [batch_size, 1] right?
        if args.train_method == "single":                                                
            pass
        else:
            # target has to be shape=[seq_len * [1,2]]
            #loss_per_digit = -tf.reduce_sum(target_list[i]*tf.log(output)) # this should be just 1 by 1 - 1 by 1            
            loss_per_digit = -(target_list[i]*tf.log(tf.clip_by_value(output, 1e-10, 1.0)) + (1-target_list[i])*tf.log(tf.clip_by_value(1-output,1e-10,1.0)))
            loss += loss_per_digit

    if args.train_method == "single":
        #output = tf.Print(output, [output], message="this is output: ")
        # print(state)
        #loss = -tf.reduce_sum(target*tf.log(output)) # this should be just 1 by 1 - 1 by 1
        loss = -(target*tf.log(tf.clip_by_value(output,1e-10,1.0)) + (1-target)*tf.log(tf.clip_by_value(1-output,1e-10,1.0)))  
        tf.scalar_summary("loss", loss)
        #train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        if args.grad_clip:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), args.max_grad_norm)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.apply_gradients(zip(grads, tvars))
        else:
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss) # 0.001
    else:
        if args.grad_clip:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), args.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.apply_gradients(zip(grads, tvars))
        else:
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    

    tf.add_to_collection('train_op', train_op)
    tf.add_to_collection('output', output)
    tf.add_to_collection('initial_state', initial_state)
    tf.add_to_collection('data', data)
    tf.add_to_collection('target', target)
    tf.add_to_collection('state', state)


    final_state = state

    summary_op = tf.merge_all_summaries()
    init_op = tf.initialize_all_variables()
    ############### Graph construction end ##########

    total_loss_list = []
    acc_list = []
    loss_diff_list = []
    saver = tf.train.Saver()
    with tf.Session() as session:
        summary_writer = tf.train.SummaryWriter("tensorflow_log", graph=session.graph)
        session.run(init_op)
        numpy_state = initial_state.eval()
        # for epoch in range(max_epoch):
        epoch = 0
        old_val_loss = 1; new_val_loss = 2*old_val_loss
        while abs(new_val_loss - old_val_loss) > args.loss_diff_eps:
            epoch += 1
            if epoch == max_epoch: 
                print("max epoch reached. break the loop:")
                break
            total_loss = 0
            step_count = 0
            for step, (x,y) in enumerate(reader.parity_iterator(input_data,target_data,batch_size, seq_len)):
                step_count += 1
                if args.train_method == "single":
                    y = getData.createTargetData(x[0])[-1]
                    y_target = np.array(y).reshape(1,1)
                    #if y == 0: y_target[0][0] = 1 
                    #else: y_target[0][1] = 1 
                else:    # y_target needs to be shape=[num_seq * [batch_size, output_size]
                    y_target = np.zeros((seq_len, 1, 1))
                    for i in range(seq_len):
                        if y[0][i] == 0: y_target[i][0][0] = 0
                        else: y_target[i][0][0] = 1
                lr_value = args.learning_rate #0.001
                if args.lr_test == True and epoch == 150:
                    lr_value = lr_value / 10

                feed_dict={initial_state: numpy_state, data: x, target: y_target, learning_rate: lr_value}
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
            print(1.0 * total_loss / step_count)
            total_loss_list.append(1.0 * total_loss[0][0] / step_count)

            acc = calc_accuracy(100, numpy_state)
            acc_list.append(acc)

            # validation (for termination criteria)
            val_total_loss = 0
            num_total_steps = 0
            for step_val, (val_x,val_y) in enumerate(reader.parity_iterator(val_input_data,val_target_data,batch_size, seq_len)):
                if args.train_method == "single": # using only the last target (at the end of the time_step)
                    val_y = getData.createTargetData(x[0])[-1] 
                    val_y_target = np.array(val_y).reshape(1,1)
                    #if val_y == 0: val_y_target[0][0] = 1 
                    #else: val_y_target[0][1] = 1
                else:
                    val_y_target = np.zeros((seq_len, 1, 1))
                    for i in range(seq_len):
                        if val_y[0][i] == 0: val_y_target[i][0][0] = 0
                        else: val_y_target[i][0][0] = 1
                val_loss = session.run([loss], feed_dict={initial_state: numpy_state, data: val_x, target: val_y_target, learning_rate: 0.0})
                val_total_loss += val_loss[0]
                num_total_steps += 1
            old_val_loss = new_val_loss
            new_val_loss = 1.0 * val_total_loss / num_total_steps
            print("printing diff")
            print(new_val_loss - old_val_loss)
            loss_diff_list.append(new_val_loss - old_val_loss)
            
            
        saver.save(session, 'my_model', global_step=0)


import pickle
with open('total_loss_list.pickle', 'wb') as f:
    pickle.dump(total_loss_list, f)

with open('acc_list.pickle', 'wb') as f:
    pickle.dump(acc_list, f)
with open('loss_diff_list.pickle', 'wb') as f:
    pickle.dump(loss_diff_list, f)

end_time = datetime.datetime.now()
end_utime = os.times()[0]                                                                                                                                                          
info['end_time'] = end_time.isoformat()
info['elapsed_time'] = str((end_time - start_time))
info['elapsed_utime'] = str((end_utime - start_utime))

with open('info.txt', 'w') as outfile:
    for key in info.keys():
        outfile.write("#%s=%s\n" % (key, str(info[key])))
