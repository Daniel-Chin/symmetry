import numpy as np
from os import path
import sys

# models = ['./model']
# CHECKPOINTS_PATHS = ['./model/checkpoint_%d.pt']
# ZLATTICE_PATHS = ['./zLattice']

model_paths = [
    '/scratch/$USER/Self-supervised-learning-via-symmetry/codes/S3Ball/dense_exp/symm_0-vae', 
]
exp_ids = [
    16, 
    # 42, 
    # 100, 
]
CHECKPOINTS_PATHS = []
ZLATTICE_PATHS = []
for model_path in model_paths:
    for exp_id in exp_ids:
        CHECKPOINTS_PATHS.append(path.join(
            model_path, str(exp_id), 'checkpoint_%d.pt', 
        ))
        ZLATTICE_PATHS.append(f'./zLattice_{exp_id}')

DATASET_PATH = './latticeDataset'

N_CURVES = 3 + 1
N_CURVE_SEGMENTS = 12
assert N_CURVE_SEGMENTS % (N_CURVES - 1) == 0
N_CURVE_VERTICES = N_CURVE_SEGMENTS + 1

X_RANGE = (-2, 2)
Y_RANGE = ( 0, 2)
Z_RANGE = ( 1, 7)

X_LATTICE = np.linspace(*X_RANGE, N_CURVE_VERTICES)
Y_LATTICE = np.linspace(*Y_RANGE, N_CURVE_VERTICES)
Z_LATTICE = np.linspace(*Z_RANGE, N_CURVE_VERTICES)
DATASET_SIZE = N_CURVE_VERTICES ** 3

IMG_W = 32
IMG_H = 32

N_LATENT_DIM = 3

CHECKPOINT_INTERVAL = 100
