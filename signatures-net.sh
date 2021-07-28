#!/bin/sh

#$ -N signatures-nets
#$ -cwd
#$ -j y
#$ -t 1-15                
#$ -q short-sl7,long-sl7
#$ -l h_rt=432000
#$ -l virtual_free=4G
#$ -o cluster/$TASK_ID.out


conda activate env_signatures

input_file=parameters.txt
input=$(awk "NR==$SGE_TASK_ID" $input_file)
echo ${input}
num_hidden_layers=$(echo ${input} | awk -F " " '{print $1}')
num_neurons=$(echo ${input} | awk -F " " '{print $2}')
initial_learning_rate=$(echo ${input} | awk -F " " '{print $3}')
learning_rate_steps=$(echo ${input} | awk -F " " '{print $4}')
learning_rate_gamma=$(echo ${input} | awk -F " " '{print $5}')
experiment_id=$(echo ${input} | awk -F " " '{print $6}')
num_mutations=$(echo ${input} | awk -F " " '{print $7}')

python3 train.py ${num_hidden_layers} ${num_neurons} ${initial_learning_rate} ${learning_rate_steps} ${learning_rate_gamma} ${experiment_id} ${num_mutations}

