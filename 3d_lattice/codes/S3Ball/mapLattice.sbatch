#!/bin/bash

#SBATCH --mail-user=daniel.chin@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=6GB
#SBATCH --job-name=lattice
#SBATCH --output=slurm_map.out
#SBATCH --error=slurm_map.err
#SBATCH --partition=aquila

module purge
module load miniconda
source activate danielTorch
python mapLattice.py
