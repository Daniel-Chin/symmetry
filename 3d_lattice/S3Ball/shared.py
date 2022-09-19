import numpy as np

DATASET_PATH = './latticeDataset'
print('DATASET_PATH =', DATASET_PATH)
print('Change?')
op = input('>')
if op != '':
    DATASET_PATH = op
CHECKPOINTS_PATH = './model/checkpoint_%d.pt'
ZLATTICE_PATH = './zLattice'

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
