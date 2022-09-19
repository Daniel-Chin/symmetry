#!/bin/bash

#SBATCH --mail-user=daniel.chin@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:01:00
#SBATCH --mem=4GB
#SBATCH --job-name=flush
#SBATCH --output=slurm_flush.out
#SBATCH --error=slurm_flush.err
#SBATCH --partition=aquila

module purge
python flushExp.py
