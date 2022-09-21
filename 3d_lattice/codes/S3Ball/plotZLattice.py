from os import path
import csv

# from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import torch

from shared import *
N_ROWS = len(expGroups) * len(RAND_INIT_IDS)

MAX_T = 100000
N_COLS = 3
EPOCH_INTERVAL = 100
PADDING = -.25

Z_SCENE_RADIUS = 3.2
COLORS = (
    (1, 0, 0), 
    (0, .7, 0), 
    (0, 0, 1), 
)

def main():
    fig = plt.figure()
    # fig, axes = plt.subplots(
    #     len(expGroups) * len(RAND_INIT_IDS), N_COLS, 
    #     projection='3d', 
    #     sharex=True, sharey=True, 
    # )
    fig.subplots_adjust(wspace=PADDING, hspace=PADDING)
    subplot_i = 0
    for exp_i, expGroup in enumerate(expGroups):
        for rand_init_i, rand_init_id in enumerate(RAND_INIT_IDS):
            row_i = exp_i * len(RAND_INIT_IDS) + rand_init_i
            for col_i in range(N_COLS):
                epoch = round(
                    (col_i + 1) / N_COLS * MAX_T / EPOCH_INTERVAL, 
                ) * EPOCH_INTERVAL
                plotOne(
                    epoch, row_i, col_i, subplot_i, fig, 
                    expGroup, rand_init_id, 
                )
                subplot_i += 1
    plt.show()
    print('ok')

def plotOne(
    epoch, row_i, col_i, subplot_i, fig: plt.Figure, 
    expGroup: ExpGroup, rand_init_id, 
):
    filename = path.join(
        expGroup.getZLatticePath(rand_init_id), 
        f'{epoch}.csv', 
    )
    # if not path.isfile(filename):
    #     print('epoch', epoch, 'not found.')
    #     raise StopIteration
    zLattice = torch.zeros((
        N_CURVE_VERTICES, 
        N_CURVE_VERTICES, 
        N_CURVE_VERTICES, 
        N_LATENT_DIM, 
    ))
    print('\r', filename
        .replace('\\', '\t\\')
        .replace('_', '\t_')
    , end=' ', flush=True)
    with open(filename, 'r') as f:
        c = csv.reader(f)
        next(c)
        for x_i in range(N_CURVE_VERTICES):
            for y_i in range(N_CURVE_VERTICES):
                for z_i in range(N_CURVE_VERTICES):
                    x_i, y_i, z_i, z0, z1, z2 = next(c)
                    x_i = int(x_i)
                    y_i = int(y_i)
                    z_i = int(z_i)
                    zLattice[x_i, y_i, z_i, 0] = float(z0)
                    zLattice[x_i, y_i, z_i, 1] = float(z1)
                    zLattice[x_i, y_i, z_i, 2] = float(z2)
        try:
            next(c)
        except StopIteration:
            pass
        else:
            raise Exception('CSV longer than expected.')

    ax = fig.add_subplot(
        N_ROWS, N_COLS, 
        subplot_i + 1, projection='3d', proj_type='ortho', 
    )
    # Look straight at the XZ plane
    ax.elev = 0
    ax.azim = 90
    ax.tick_params(
        'both', which='both', 
        # bottom=False, 
        # top=False, 
        # left=False, 
        # right=False,
        labelbottom = row_i == N_ROWS - 1, 
        labelleft = col_i == 0, 
    )
    ax.set_xlim(-Z_SCENE_RADIUS, Z_SCENE_RADIUS)
    ax.set_ylim(-Z_SCENE_RADIUS, Z_SCENE_RADIUS)
    ax.set_zlim(-Z_SCENE_RADIUS, Z_SCENE_RADIUS)
    ax.set_xticks([-2.5, 0, 2.5])
    ax.set_yticks([])
    ax.set_zticks([-2.5, 0, 2.5])

    step = N_CURVE_SEGMENTS // (N_CURVES - 1)
    for curve_i in range(0, N_CURVE_SEGMENTS, step):
        # print('curve_i', curve_i)
        for curve_j in range(0, N_CURVE_SEGMENTS, step):
            z_segs = (
                zLattice[:, curve_i, curve_j, :], 
                zLattice[curve_i, :, curve_j, :], 
                zLattice[curve_i, curve_j, :, :], 
            )
            for z_seg, color in zip(z_segs, COLORS):
                z_seg = z_seg.numpy()
                ax.plot(
                    z_seg[:, 0], 
                    z_seg[:, 1], 
                    z_seg[:, 2], 
                    c=color, 
                )

main()
