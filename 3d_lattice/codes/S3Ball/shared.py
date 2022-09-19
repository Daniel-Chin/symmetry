import numpy as np
from os import path
import sys

DATASET_PATH = './latticeDataset'
try:
    _, model_path, exp_id = sys.argv
    CHECKPOINTS_PATH = path.join(
        model_path, exp_id, 'checkpoint_%d.pt', 
    )
    ZLATTICE_PATH = f'./zLattice_{exp_id}'
except ValueError:
    model_path = './model'
    CHECKPOINTS_PATH = './model/checkpoint_%d.pt'
    ZLATTICE_PATH = './zLattice'
sys.path.append(path.abspath(model_path))

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
