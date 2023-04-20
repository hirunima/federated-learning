#!/bin/bash
#SBATCH --job-name=fedavg
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --partition=dpart
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00

#SBATCH --output=hiruOut_type.txt
#SBATCH --error=hiruError_type.txt

source /vulcanscratch/hirunima/anaconda3/etc/profile.d/conda.sh
# conda env list
conda activate fed_learn

python src/federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=1000
