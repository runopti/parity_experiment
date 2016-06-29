import argparse
import sys
parser = argparse.ArgumentParser(description='Check num of graphs to plot for Parity Experiment.')

parser.add_argument('--num_plot', type=int)
#print(sys.argv[2]) 
#if sys.args[0] == 1:
#    print("ok")

#parser = argparse.ArgumentParser(description='Define parameters for Parity Experiment.')
parser.add_argument('--max_epoch', type=int)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--output_size', type=int)
parser.add_argument('--seq_len', type=int)
parser.add_argument('--data_size', type=int)
parser.add_argument('--rseed', type=int)

parser.add_argument('--yaxis', nargs=int(sys.argv[2]), type=str)
parser.add_argument('--xaxis', nargs=int(sys.argv[2]), type=str)
num_plot = int(sys.argv[2])
print(int(sys.argv[2]))

args = parser.parse_args()

print(args.yaxis)
print(args.yaxis[0].split(".")[1])

max_epoch = args.max_epoch  # 300
hidden_size = args.hidden_size # 2 # 4
batch_size = args.batch_size #1
output_size = args.output_size  #2
seq_len = args.seq_len  #12 #3  #6 

n = args.data_size # 1000
print(type(max_epoch))
#np.random.seed(args.rseed)
#tf.set_random_seed(args.seed)

import numpy as np
import matplotlib.pyplot as plt


NSAMPLE = 1000
x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
r_data = np.float32(np.random.normal(size=(NSAMPLE,1)))
y_data = np.float32(np.sin(0.75*x_data)*7.0+x_data*0.5+r_data*1.0)

image_name = "sig_function"
plt.figure(figsize=(8, 8))
plot_out = plt.plot(x_data,y_data,'ro',alpha=0.3)
#plt.savefig(image_name + '.png')

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


for i in args.__dict__.keys():
    if i[0] != "_":
        info['option_' + i] = str(getattr(args, i))



end_time = datetime.datetime.now()
end_utime = os.times()[0]
info['end_time'] = end_time.isoformat()
info['elapsed_time'] = str((end_time - start_time))
info['elapsed_utime'] = str((end_utime - start_utime))

with open('info.txt', 'w') as outfile:
    for key in info.keys():
        outfile.write("#%s=%s\n" % (key, str(info[key])))
