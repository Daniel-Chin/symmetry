from os import path
import csv

from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
import torch

from shared import *
N_ROWS = len(expGroups) * len(RAND_INIT_IDS)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
})
DPI = plt.rcParams['figure.dpi']

MAX_T = 80000
N_COLS = 15
EPOCH_INTERVAL = 100

Z_SCENE_RADIUS = 3.2

MYSTERIOUS_RATIO = .035
FIGSIZE = (7, 5)
FONTSIZE = 12
HSPACE = .2
TITLE_HSPACE = 1.8
TITLE_HSPACE_PER_ROW = TITLE_HSPACE / len(RAND_INIT_IDS)
COLORS = (
    (.9, 0, 0), 
    (0, .7, 0), 
    (0, 0, 1), 
)

def SubplotIter():
    for col_i in range(N_COLS):
        batch_i = round(
            (col_i + 1) / N_COLS * MAX_T / EPOCH_INTERVAL, 
        ) * EPOCH_INTERVAL
        for exp_i, expGroup in enumerate(expGroups):
            for rand_init_i, rand_init_id in enumerate(RAND_INIT_IDS):
                row_i = exp_i * len(RAND_INIT_IDS) + rand_init_i
                yield (
                    batch_i, row_i, col_i, expGroup, 
                    rand_init_i, rand_init_id, 
                )

def main():
    fig = plt.figure(figsize=FIGSIZE)
    axes = fig.subplots(
        len(expGroups) * len(RAND_INIT_IDS), N_COLS, 
        sharex=True, sharey=True, 
    )
    fig.subplots_adjust(
        hspace = HSPACE + TITLE_HSPACE_PER_ROW, 
    )
    for (
        batch_i, row_i, col_i, expGroup, 
        rand_init_i, rand_init_id, 
    ) in SubplotIter():
        ax = axes[row_i][col_i]
        plotOne(
            batch_i, row_i, col_i, ax, expGroup, rand_init_id, 
        )
        if rand_init_i == 0 and col_i == N_COLS // 2:
            ax.set_title(expGroup.display)

    bBoxes = []
    for (
        batch_i, row_i, col_i, expGroup, 
        rand_init_i, rand_init_id, 
    ) in SubplotIter():
        ax = axes[row_i][col_i]
        y_offset = rand_init_i * TITLE_HSPACE_PER_ROW
        y_offset *= MYSTERIOUS_RATIO
        bBox = ax.get_position()
        bBoxes.append(Bbox([
            [bBox.x0, bBox.y0 + y_offset], 
            [bBox.x1, bBox.y1 + y_offset], 
        ]))
        if row_i == N_ROWS - 1:
            epoch = str(round(batch_i / 16))
            # ax.set_xlabel((
            #     'Epoch = ' if col_i == 0 else ''
            # ) + epoch)
            # ax.set_xlabel(f'Epoch {epoch}')

            EPOCH_SINK = -2 * DPI
            ax.annotate(
                epoch, xy=(
                    (
                        bBox.width * DPI / MYSTERIOUS_RATIO
                    ) / 2 - textWidth(epoch) / 2, 
                    EPOCH_SINK, 
                ), 
                xycoords='axes pixels', fontsize=FONTSIZE, 
            )
            if col_i == 0:
                EPOCH_HEADING = 'Epoch '
                ARROW_SINK = -1.3 * DPI
                ARROW_X = 0 * DPI
                ax.annotate(
                    EPOCH_HEADING, xy=(
                        ARROW_X - textWidth(EPOCH_HEADING), 
                        ARROW_SINK - textHeight() / 2, 
                    ), 
                    xycoords='axes pixels', fontsize=FONTSIZE, 
                )
                ax.annotate(
                    '', 
                    xytext=(
                        ARROW_X, ARROW_SINK, 
                    ), 
                    xy=(
                        (
                            DPI * 26 
                            - bBox.x0 * DPI / MYSTERIOUS_RATIO
                        ), 
                        ARROW_SINK, 
                    ), 
                    xycoords='axes pixels', arrowprops=dict(
                        arrowstyle='-|>', 
                        shrinkA=0, 
                        shrinkB=0, 
                    ), 
                )
    for (
        batch_i, row_i, col_i, expGroup, 
        rand_init_i, rand_init_id, 
    ) in SubplotIter():
        ax = axes[row_i][col_i]
        ax.set_position(bBoxes.pop(0))
    plt.show()
    print('ok')

def plotOne(
    batch_i, row_i, col_i, ax: plt.Axes, 
    expGroup: ExpGroup, rand_init_id, 
):
    filename = path.join(
        expGroup.getZLatticePath(rand_init_id), 
        f'{batch_i}.csv', 
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

    ax.set_xlim(-Z_SCENE_RADIUS, Z_SCENE_RADIUS)
    ax.set_ylim(-Z_SCENE_RADIUS, Z_SCENE_RADIUS)
    # ax.set_xticks([-2, 0, 2])
    # ax.set_yticks([-2, 0, 2])
    ax.set_xticks([-2, 2])
    ax.set_yticks([-2, 2])

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
                    z_seg[:, 2], 
                    c=color, linewidth=.5, 
                )

def textHeight():
    return FONTSIZE * 2.8

def textWidth(x):
    return len(x) * textHeight()

main()
