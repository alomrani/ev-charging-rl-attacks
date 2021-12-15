#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=03:00:00
#SBATCH --output=./%N-%j.out

#GAMMA=$1

module load python/3.7
module load scipy-stack

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index seaborn torch scipy numpy tqdm matplotlib

python train_dnn.py --n_epochs 200 --batch_size 1000 --lr_model 0.0002 --lr_decay 1.0 --p 0.6
# python train_dnn.py --n_epochs 200 --batch_size 1000 --lr_model 0.0004 --lr_decay 1.0 --p 0.75 --train_dataset dnn_datasets/detection_train.pt --val_dataset dnn_datasets/detection_val.pt --test_dataset dnn_datasets/detection_test.pt 
#python train_rl.py --batch_size 1072 --num_cars 30 --num_epochs 200 --train_seed --regularize --gamma $GAMMA --lr_model 0.00004 --lr_decay 0.99 --exp_beta 0.75
