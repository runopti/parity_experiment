import tensorflow as tf
from tensorflow.python.ops import rnn_cell

init_scale = 0.1
num_steps = 7
num_units = 7
input_data = [1, 2, 3, 4, 5, 6, 7]
target = [2, 3, 4, 5, 6, 7, 7]

batch_size = 1

with tf.Graph().as_default(), tf.Session() as session:
    
    input1 = tf.placeholder(tf.float32, [batch_size, 1])
    inputs = [input1 for _ in range(num_steps)]

    lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    initial_state = state = tf.zeros([batch_size, 2*num_unites])
    loss = tf.constant(0.0)

    # unroll
    for i in range(num_steps):
        if i > 0: tf.get_variable_scope().reuse_variables()        
        cell_output, state = lstm(inputs[i], state)
        loss += tf.reduce_sum(abs(cell_output - target))
    final_state = state

    
    optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)
    max_grad_norm = 5
    trainable_variables = tf.trainable_variables()

    numpy_state = initial_state.eval()
    session.run(tf.initialize_all_variables())
    for epoch in range(10):
        for k in range(7): 

