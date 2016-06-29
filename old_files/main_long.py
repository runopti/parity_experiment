from models import rnn_long
import tensorflow as tf
import numpy as np
import getData 
import reader

np.random.seed(0)
tf.set_random_seed(0)

n = 1000
batch_size = 1
input_data = getData.createInputData(n)
target = getData.createTargetData(input_data)

#input_data = np.asarray(input_data).reshape(batch_size,n)
#print(input_data.shape)
#target = np.asarray(target).reshape(batch_size,n)
#print(target)
#print(target[0][1].shape)

#mlp = model.MLP(10, 10, 3, 3, 1)
#weights = mlp.test()

batch_size = 1
seq_len = 20
target_size = 20
hidden_size = 4

def see_output(data_size):
    test_input = getData.createInputData(data_size)
    test_target = getData.createTargetData(test_input)

    print("printing test_input")
    print(test_input)
    print("printing test_target")
    print(test_target)
    print("printing model output")
    print(sess.run(m.output_op, feed_dict={m.input: np.array(test_input).reshape(1,data_size), m.target: np.array(test_target).reshape(1,data_size)}))

m = rnn_long.RNNmodel(seq_len, target_size, hidden_size, batch_size)
num_epoch = 40
#init_op = tf.initialize_all_variables()
init_op = m.get_init_op
summary_op = m.get_summary_op
#exit()
with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter('logdata', graph=sess.graph)
    sess.run(init_op)
    #print(sess.run(weights))
    for i in range(num_epoch):
        state = m.initial_state.eval()
        for step, (x,y) in enumerate(reader.parity_iterator(input_data, target, batch_size, seq_len)):
            cost, state, _ = sess.run([m.loss, m.final_state, m.train_op], 
                                      {m.input: x,
                                       m.target: y,
                                       m.initial_state:state})
            if step % 20 == 0:
                print("printing loss")
                print(sess.run(m.loss, feed_dict={m.input: x, m.target: y, m.initial_state: state}))
        summary_str=sess.run(summary_op, feed_dict={m.input: x, m.target: y})
        summary_writer.add_summary(summary_str, i)
         
    see_output(20)



