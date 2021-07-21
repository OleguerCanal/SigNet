#!/bin/sh

#$ -N finetuner_optimizer
#$ -cwd
#$ -j y
#$ -t 1-100              
#$ -q short-sl7,long-sl7
#$ -l h_rt=432000
#$ -l virtual_free=4G
#$ -o Cluster/$TASK_ID.out

input_file=finetuner_IDs.txt
input=$(awk "NR==$SGE_TASK_ID" $input_file)
echo ${input}
ID=$(echo ${input} | awk -F " " '{print $1}')
echo ${ID}

python optimize_errorfinder.py ${ID}