import numpy as np
import math
np.random.seed(0)

def createTargetData(digit_list):
    """ generate len(digit_list) vector, whose i th element is the
    corresponding parity upto the ith element of the input."""
    sum_ = 0
    parity_list = []

    # check the first digit
    sum_ = digit_list[0]
    if sum_ % 2 == 0:
        parity_list.append(0)
    else:
        parity_list.append(1)

    # main for loop
    for i in range(len(digit_list)):
        if i == 0: 
            continue
        else:
            sum_ += digit_list[i]
            if sum_ % 2 == 0:
                parity_list.append(0)
            else:
                parity_list.append(1)

    return parity_list

def createInputData(n):
    """ generate a list of digits 0/1"""
    digit_list = []
    for i in range(n):
        digit_list.append(np.random.randint(2))
    return digit_list

def createRandomData(n, n_test, seq_len):
    """ uniformly sample a data point from the 2^n space. """
    data_input_list = []
    data_test_list = []
    store_int_list = []
    for _ in range(n):
        random_int = np.random.randint(math.pow(2,seq_len)) # sample a number from [1, 2^seq_len]
        store_int_list.append(random_int)
        data_input_list.append(np.array(list(bin(random_int)[2:].zfill(seq_len))).astype(float)) # convert this into binary digit
    for _ in range(n_test):
        random_int = store_int_list[0] # just to pass the first test in while
        while random_int in store_int_list:
            random_int = np.random.randint(math.pow(2,seq_len))
        data_test_list.append(np.array(list(bin(random_int)[2:].zfill(seq_len))).astype(float))
        
    return data_input_list, data_test_list

#data = createInputData(10)
#parity = createTargetData(data)

#print(data)
#print(parity)
