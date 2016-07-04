#!/bin/sh
## Experiment1 
filename="experiment1-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_test_classif_graph.py --max_epoch 600 --hidden_size 3 --batch_size 1 --output_size 2 --seq_len 5  --data_size 1000 --rseed 0
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 

# th ${script_dir_path}/../../src/plot_table.lua -i norm_gradParam.bin  -name ${image_name} --save 'img'  -xlabel "number of update" -ylabel "norm of gradient" 
# plot the norm_gradParam.bin and name it as plot-the-date-time-it-was-created.png. Saved in img folder.

#sed -e "1c\test_acc" ${script_dir_path}/$filename/logs/test.log > ${script_dir_path}/$filename/logs/test.csv 
#sed -e "1c\train_acc" ${script_dir_path}/$filename/logs/train.log > ${script_dir_path}/$filename/logs/train.csv

#image_name="epochPlot-`date \"+%F-%T"`.png"

#th ${script_dir_path}/../../src/plot_table.lua -epochPlot -xlabel "epoch" -ylabel "accuracy" -input1 ${script_dir_path}/$filename/logs/train.csv -input2 ${script_dir_path}/$filename/logs/test.csv  -name ${image_name} --save 'img'

python3 ${script_dir_path}/../../src/plot_table.py --num_plot 2 --save img --max_epoch 600 --yaxis acc_list.pickle total_loss_list.pickle --xaxis 600 600


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../

## Experiment 2

filename="experiment2-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_test_classif_graph.py --max_epoch 600 --hidden_size 3 --batch_size 1 --output_size 2 --seq_len 5 --data_size 2000 --rseed 0
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 2 --save img --max_epoch 600 --yaxis acc_list.pickle total_loss_list.pickle --xaxis 600 600 


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../

## Experiment 3

filename="experiment3-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_test_classif_graph.py --max_epoch 600 --hidden_size 3 --batch_size 1 --output_size 2 --seq_len 5 --data_size 4000 --rseed 0
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 2 --save img --max_epoch 600 --yaxis acc_list.pickle total_loss_list.pickle --xaxis 600 600 


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../

## Experiment 4

filename="experiment4-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_test_classif_graph.py --max_epoch 600 --hidden_size 4 --batch_size 1 --output_size 2 --seq_len 5 --data_size 8000 --rseed 0
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 2 --save img --max_epoch 600 --yaxis acc_list.pickle total_loss_list.pickle --xaxis 600 600 


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../
