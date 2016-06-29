import model
import tensorflow as tf
import numpy as np

input_data = np.linspace(-1,1,10)
input_data = input_data.reshape(1,10)
print(input_data.shape)
target = np.arange(10)
target = target.reshape(1,10)
print(target)
print(target[0][1].shape)
mlp = model.MLP(10, 10, 3, 3, 1)
weights = mlp.test()

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    #print(sess.run(weights))
    print(sess.run(mlp.output_op, feed_dict={mlp.input_data: input_data, mlp.target: [target[0][1]]}))
