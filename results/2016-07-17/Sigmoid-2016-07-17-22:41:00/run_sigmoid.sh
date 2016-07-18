#!/bin/sh
## Experiment 1

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 50 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "single" --lr_test True --cell_type "lstm" --learning_rate 0.001
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../


## Experiment 2

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 60 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "single" --lr_test True --cell_type "lstm" --learning_rate 0.001                                                              
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../ 

## Experiment 3

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 70 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "single" --lr_test True --cell_type "lstm" --learning_rate 0.001                                                              
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../ 

## Experiment 4

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 80 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "single" --lr_test True --cell_type "lstm" --learning_rate 0.001                                                              
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../ 


## Experiment 5

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 90 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "single" --lr_test True --cell_type "lstm" --learning_rate 0.001                                                              
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../ 

## Experiment 6

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 100 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "single" --lr_test True --cell_type "lstm" --learning_rate 0.001                                                              
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../ 


#### Start "total" instead of "single"
## Experiment 1

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 50 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "total" --lr_test True --cell_type "lstm" --learning_rate 0.001
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../


## Experiment 2

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 60 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "total" --lr_test True --cell_type "lstm" --learning_rate 0.001                                                              
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../ 

## Experiment 3

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 70 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "total" --lr_test True --cell_type "lstm" --learning_rate 0.001                                                              
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../ 

## Experiment 4

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 80 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "total" --lr_test True --cell_type "lstm" --learning_rate 0.001                                                              
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../ 


## Experiment 5

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 90 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "total" --lr_test True --cell_type "lstm" --learning_rate 0.001                                                              
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../ 

## Experiment 6

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 100 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "total" --lr_test True --cell_type "lstm" --learning_rate 0.001                                                              
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../ 


## Experiment 7

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 125 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "total" --lr_test True --cell_type "lstm" --learning_rate 0.001
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../


## Experiment 2

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 150 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "total" --lr_test True --cell_type "lstm" --learning_rate 0.001                                                              
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../ 

## Experiment 3

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 175 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "total" --lr_test True --cell_type "lstm" --learning_rate 0.001                                                              
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../ 

## Experiment 4

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 200 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "total" --lr_test True --cell_type "lstm" --learning_rate 0.001                                                              
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../ 


## Experiment 5

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 225 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "total" --lr_test True --cell_type "lstm" --learning_rate 0.001                                                              
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../ 

## Experiment 6

filename="Sigmoid-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_simple_sigm_data.py --max_epoch 500 --hidden_size 250 --batch_size 1 --output_size 1 --seq_len 13 --data_size 4000 --rseed 0  --loss_diff_eps 1e-8 --grad_clip True --max_grad_norm 1 --train_method "total" --lr_test True --cell_type "lstm" --learning_rate 0.001                                                              
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 3 --save img  --yaxis acc_list.pickle total_loss_list.pickle loss_diff_list.pickle


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../ 



