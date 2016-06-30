#!/bin/sh
## Experiment1 
filename="experiment1-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

#th ${script_dir_path}/../../src/train_cifar.lua -batchSize 50000 -maxEpoch 500 -full -hessian -lineSearch  -currentDir ${script_dir_path}/../../src -gradNormThresh 0.1 -hessianMultiplier 1 -iterMethodDelta 10e-10 -iterationMethod lanczos -modelpath /models/cifar_mlp_10x10.lua -shrink
python3 ${script_dir_path}/../../src/rnn_test_classif_graph.py --max_epoch 300 --hidden_size 1 --batch_size 1 --output_size 2 --seq_len 3  --data_size 1000
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

python3 ${script_dir_path}/../../src/plot_table.py --num_plot 2 --save img --max_epoch 300 --yaxis acc_list.pickle total_loss_list.pickle --xaxis 300 300


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../

## Experiment 2

filename="experiment2-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_test_classif_graph.py --max_epoch 300 --hidden_size 1 --batch_size 1 --output_size 2 --seq_len 4 --data_size 1000
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 2 --save img --max_epoch 300 --yaxis acc_list.pickle total_loss_list.pickle --xaxis 300 300 


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../

## Experiment 3

filename="experiment3-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_test_classif_graph.py --max_epoch 300 --hidden_size 1 --batch_size 1 --output_size 2 --seq_len 5 --data_size 1000
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 2 --save img --max_epoch 300 --yaxis acc_list.pickle total_loss_list.pickle --xaxis 300 300 


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../

## Experiment 4

filename="experiment4-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

python3 ${script_dir_path}/../../src/rnn_test_classif_graph.py --max_epoch 300 --hidden_size 1 --batch_size 1 --output_size 2 --seq_len 6 --data_size 1000
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="plot-`date \"+%F-%T"`.png" 


python3 ${script_dir_path}/../../src/plot_table.py --num_plot 2 --save img --max_epoch 300 --yaxis acc_list.pickle total_loss_list.pickle --xaxis 300 300 


cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

cd ../
