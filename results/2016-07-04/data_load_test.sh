#!/bin/sh
## Experiment 3

filename="lr_schedule_test-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/test.py --max_epoch 600 --hidden_size 3 --batch_size 1 --output_size 2 --seq_len 5 --data_size 2000 --rseed 0 --diff_loss_eps 1e-3
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 2 --save img  --yaxis acc_list.pickle total_loss_list.pickle 


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../

