import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

test = [i for i in range(100)]

import pickle
with open("test.pickle", "wb") as file:
    pickle.dump(test, file)



