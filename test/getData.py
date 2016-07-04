import numpy as np

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

#data = createInputData(10)
#parity = createTargetData(data)

#print(data)
#print(parity)
