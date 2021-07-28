#!/bin/sh

#$ -N signatures-nets
#$ -cwd
#$ -pe smp 16                
#$ -q short-sl7,long-sl7
#$ -l h_rt=432000
#$ -l virtual_free=4G
#$ -o cluster/train.out


conda activate env_signatures


python3 baseline.py train

