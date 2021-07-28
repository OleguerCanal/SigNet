#!/bin/sh

#$ -N finetuner_optimizer
#$ -cwd
#$ -q short-sl7,long-sl7
#$ -l h_rt=432000
#$ -l virtual_free=4G
#$ -o Cluster/search.out

conda activate sigs_env
python optimize_finetuner.py
