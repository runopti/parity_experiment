import tensorflow as tf
import numpy as np

def parity_iterator(raw_data, raw_target, batch_size, num_steps):
  """Iterate on the raw PTB data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.                       
  """
  raw_data = np.array(raw_data, dtype=np.float32)
  raw_target = np.array(raw_target, dtype=np.float32)

  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.float32)
  target = np.zeros([batch_size, batch_len], dtype=np.float32)
  if batch_size > 1:
    for i in range(batch_size):
      data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
      target[i] = raw_target[batch_len * i:batch_len * (i + 1)]
  else:
    data[0,:] = raw_data
    target[0,:] = raw_target


  epoch_size = (batch_len - 1) // num_steps
  print(len(raw_data))
  print(batch_len)

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  #if batch_size > 1:
  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = target[:, i*num_steps:(i+1)*num_steps]
    yield (x, y)
  #else:
   # for i in range(epoch_size):
    #  x = data[i*num_steps:(i+1)*num_steps]
     # y = target[i*num_steps:(i+1)*num_steps]
      #yield (x, y)
      

