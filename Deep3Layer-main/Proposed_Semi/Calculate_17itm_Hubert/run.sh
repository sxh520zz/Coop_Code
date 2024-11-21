#!/bin/bash
#SBATCH -n 1 # 指定核心数量
#SBATCH -N 1 # 指定node的数量
#SBATCH --gres=gpu:1 # 需要使用多少GPU，n是需要的数量
#SBATCH --exclude mrcompute01,compute02,compute01
#SBATCH -t 1-00:00:00


#nohup python -u train.py > 1.log 2>&1 &
python train.py