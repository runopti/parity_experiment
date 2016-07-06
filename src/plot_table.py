import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle 
import sys

import argparse
parser = argparse.ArgumentParser(description='Define parameters for plot_table.py.')
parser.add_argument('--num_plot', type=int)
num_plot = int(sys.argv[2])

parser.add_argument('--save', type=str)
#parser.add_argument('--max_epoch', type=int)

parser.add_argument('--yaxis', nargs=int(sys.argv[2]), type=str)
#parser.add_argument('--xaxis', nargs=int(sys.argv[2]), type=str)

args = parser.parse_args()

max_epoch = args.max_epoch

plot_list = []
for i in range(num_plot):
    with open(args.yaxis[i],'rb') as f:
        plot_list.append(pickle.load(f))

for i in range(num_plot):
    plt.figure(figsize=(8, 8)) 
    plot_out = plt.plot(range(len(plot_list)), plot_list[i] ,'ro',alpha=0.3)
    #plt.show()
    image_file = args.save + '/' + args.yaxis[i].split('.')[0] # taking the name before 'dot' pickle 
    plt.savefig(image_file + ".png")


#plt.figure(figsize=(8, 8)) 
#plot_out = plt.plot(range(max_epoch),acc_list,'bo', alpha=0.3)
#plt.show()
#image_file = args.save + '/acc_list'
#plt.savefig(image_file + ".png")
