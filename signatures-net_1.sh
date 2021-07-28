#!/bin/sh

#$ -N signatures-nets
#$ -cwd             
#$ -q short-sl7,long-sl7
#$ -l h_rt=432000
#$ -l virtual_free=64G
#$ -o cluster/$TASK_ID.out


conda activate env_signatures

python3 train.py

