import numpy as np

DATASET_PATH = './latticeDataset'
X_LATTICE = np.linspace(-1, 1, 4)
Y_LATTICE = np.linspace( 0, 2, 4)
Z_LATTICE = np.linspace(0, 2, 4)
X_LATTICE_LEN = len(X_LATTICE)
Y_LATTICE_LEN = len(Y_LATTICE)
Z_LATTICE_LEN = len(Z_LATTICE)
DATASET_SIZE = X_LATTICE_LEN * Y_LATTICE_LEN * Z_LATTICE_LEN

IMG_W = 32
IMG_H = 32
