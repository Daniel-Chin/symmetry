#!/bin/bash

#SBATCH --mail-user=nq285@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=6GB
#SBATCH --gres=gpu:2080Ti:1
#SBATCH --job-name=symm_4-ae
#SBATCH --output=slurm_train.out
#SBATCH --error=slurm_train.err
#SBATCH --partition=aquila,gpu

module purge
module load miniconda
source activate danielTorch


cd /gpfsnyu/scratch/nq285/Self-supervised-learning-via-symmetry/codes/S3Ball/dense_exp/symm_4-ae/
python main_train.py


