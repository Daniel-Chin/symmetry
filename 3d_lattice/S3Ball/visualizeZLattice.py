from os import path
from itertools import count
import csv

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch

from shared import *

CELL_RESOLUTION = 256
HEADING_ROW_HEIGHT = 0.3
Z_SCENE_RADIUS = 3

FONT = ImageFont.truetype("verdana.ttf", 24)
DEFAULT_PERSPECTIVE = torch.Tensor([
    [1, 0, .5], 
    [0, 1, .5], 
])

RAND_INIT_TIMES = 1
EXPERIMENTS = ['cam 0', 'cam 1']

def main():
    frame_width_height = (
        CELL_RESOLUTION * (len(EXPERIMENTS)), 
        round(CELL_RESOLUTION * (
            HEADING_ROW_HEIGHT + RAND_INIT_TIMES
        )), 
    )
    vidOut = cv2.VideoWriter(
        'zLattice.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
        5, frame_width_height, 
    )
    try:
        for epoch in count(CHECKPOINT_INTERVAL, CHECKPOINT_INTERVAL):
            visualizeOneEpoch(
                epoch, vidOut, frame_width_height, 
            )
    except StopIteration:
        print('total epoch', epoch)
    vidOut.release()
    print('written to MP4.')

def visualizeOneEpoch(epoch, vidOut, frame_width_height):
    frame = Image.new('RGB', frame_width_height)
    imDraw = ImageDraw.Draw(frame)
    textCell(
        imDraw, f'{epoch=}', 1, HEADING_ROW_HEIGHT * .2, 
    )
    for exp_i, exp_name in enumerate(EXPERIMENTS):
        textCell(
            imDraw, exp_name, 
            exp_i + .5, HEADING_ROW_HEIGHT * .5, 
        )
        for rand_init_i in range(RAND_INIT_TIMES):
            filename = path.join(ZLATTICE_PATH, f'{epoch}.csv')
            if not path.isfile(filename):
                print('epoch', epoch, 'not found.')
                raise StopIteration
            zLattice = torch.zeros((
                N_CURVE_VERTICES, 
                N_CURVE_VERTICES, 
                N_CURVE_VERTICES, 
                N_LATENT_DIM, 
            ))
            print(filename)
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
            # temp rotation
            if exp_i == 0:
                perspective = torch.Tensor([
                    [1, 0, .5], 
                    [0, 1, .5], 
                ])
            else:
                perspective = torch.Tensor([
                    [.5, 1, 0], 
                    [.5, 0, 1], 
                ])
            drawCell(
                imDraw, zLattice, exp_i, 
                rand_init_i + HEADING_ROW_HEIGHT, 
                perspective, 
            )
    vidOut.write(cv2.cvtColor(
        np.asarray(frame), cv2.COLOR_BGR2RGB, 
    ))

def textCell(imDraw, text, col_i, row_i):
    imDraw.text((
        col_i * CELL_RESOLUTION, 
        row_i * CELL_RESOLUTION, 
    ), text, font=FONT, anchor='mm')

def drawCell(
    imDraw, zLattice: torch.Tensor, col_i, row_i, 
    perspective=DEFAULT_PERSPECTIVE, 
):
    x_offset = CELL_RESOLUTION * col_i
    y_offset = CELL_RESOLUTION * row_i
    step = N_CURVE_SEGMENTS // (N_CURVES - 1)
    for curve_i in range(0, N_CURVE_SEGMENTS, step):
        # print('curve_i', curve_i)
        for curve_j in range(0, N_CURVE_SEGMENTS, step):
            z_segs = (
                zLattice[:, curve_i, curve_j, :], 
                zLattice[curve_i, :, curve_j, :], 
                zLattice[curve_i, curve_j, :, :], 
            )
            for z_seg, color in zip(z_segs, ('red', 'green', 'blue')):
                coords = []
                for i in range(N_CURVE_VERTICES):
                    x, y = perspective @ z_seg[i, :]
                    x = x.item()
                    y = y.item()
                    coords.append((
                        rasterize(
                            x, Z_SCENE_RADIUS, CELL_RESOLUTION, 
                        ) + x_offset, 
                        rasterize(
                            y, Z_SCENE_RADIUS, CELL_RESOLUTION, 
                        ) + y_offset, 
                    ))
                imDraw.line(coords, color)

def rasterize(x, x_radius, resolution):
    return round((x + x_radius) / (
        x_radius * 2
    ) * resolution)

main()
