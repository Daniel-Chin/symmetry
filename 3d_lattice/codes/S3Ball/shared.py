import numpy as np
from os import path

class ExpGroup:
    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path
    
    def getCheckpoint(self, rand_init_id, epoch):
        return path.join(
            self.model_path, 
            rand_init_id, 
            f'checkpoint_{epoch}.pt', 
        )
    
    def getZLatticePath(self, rand_init_id):
        x = './zLattice'
        if rand_init_id == '':
            return x
        else:
            return x + '_' + str(rand_init_id)

# expGroups = [ExpGroup('xjMethod', './model')]
# RAND_INIT_IDS = ['']

expGroups = [
    ExpGroup('vae aug 0', path.expandvars('/scratch/$USER/Self-supervised-learning-via-symmetry/codes/S3Ball/dense_exp/symm_0-vae')), 
    # ExpGroup('cam 0', path.expandvars('/scratch/$USER/Self-supervised-learning-via-symmetry/codes/S3Ball/dense_exp/symm_0-vae')), 
    # ExpGroup('cam 1', path.expandvars('/scratch/$USER/Self-supervised-learning-via-symmetry/codes/S3Ball/dense_exp/symm_0-vae')), 
    # ExpGroup('cam 2', path.expandvars('/scratch/$USER/Self-supervised-learning-via-symmetry/codes/S3Ball/dense_exp/symm_0-vae')), 
]
RAND_INIT_IDS = [
    16, 
    42, 
    100, 
]

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
