import sys
from shared import *
assert len(model_paths) == 1
sys.path.append(path.abspath(model_paths[0]))

import os
import shutil
import csv

import torch
from torch import Tensor
from PIL import Image

from normal_rnn import Conv2dGruConv2d, LAST_CN_NUM, LAST_H, LAST_W, IMG_CHANNEL
from train_config import CONFIG

device = torch.device("cpu")

def loadNNs(checkpoints_path):
    i = 0
    while True:
        i += CHECKPOINT_INTERVAL
        checkpoint_path = checkpoints_path % i
        if os.path.isfile(checkpoint_path):
            model = Conv2dGruConv2d(CONFIG).to(device)
            model.eval()
            model.load_state_dict(torch.load(
                checkpoint_path, 
                map_location=device, 
            ))
            yield model
        else:
            return

def loadDataset():
    prev_cwd = os.getcwd()
    os.chdir(DATASET_PATH)
    dataset = torch.zeros((
        N_CURVE_VERTICES, 
        N_CURVE_VERTICES, 
        N_CURVE_VERTICES, 
        3, 
        IMG_W, 
        IMG_H, 
    ))
    for x_i in range(N_CURVE_VERTICES):
        for y_i in range(N_CURVE_VERTICES):
            for z_i in range(N_CURVE_VERTICES):
                filename = f'{x_i}_{y_i}_{z_i}.png'
                img = Image.open(filename)
                torchImg = img2Tensor(img)
                for c in range(3):
                    dataset[
                        x_i, y_i, z_i, c, :, :
                    ] = torchImg[:, :, c]
    os.chdir(prev_cwd)
    return dataset.to(device)

def img2Tensor(img):
    np_img = np.asarray(img)
    return (
        torch.from_numpy(np_img / 256).float()
    )

def main():
    print('load dataset...', flush=True)
    dataset = loadDataset()
    for checkpoints_path, zlattice_path in zip(
        CHECKPOINTS_PATHS, ZLATTICE_PATHS, 
    ):
        print(zlattice_path, flush=True)
        nns = loadNNs(checkpoints_path)
        shutil.rmtree(zlattice_path, ignore_errors=True)
        os.makedirs(zlattice_path, exist_ok=True)
        for i, nn in enumerate(nns):
            t = (i + 1) * CHECKPOINT_INTERVAL
            print('\r epoch', t, end=' ', flush=True)
            with torch.no_grad():
                out: Tensor = nn.encoder(dataset.view(
                    DATASET_SIZE, 3, IMG_W, IMG_H, 
                ))
                mu: Tensor = nn.fc11(out.view(DATASET_SIZE, -1))
                zLattice = mu.view(
                    N_CURVE_VERTICES, 
                    N_CURVE_VERTICES, 
                    N_CURVE_VERTICES, 
                    N_LATENT_DIM, 
                )
            with open(os.path.join(
                zlattice_path, f'{t}.csv', 
            ), 'w', newline='') as f:
                c = csv.writer(f)
                c.writerow(['x_i', 'y_i', 'z_i', 'z0', 'z1', 'z2'])
                for x_i in range(N_CURVE_VERTICES):
                    for y_i in range(N_CURVE_VERTICES):
                        for z_i in range(N_CURVE_VERTICES):
                            z = zLattice[x_i, y_i, z_i, :]
                            c.writerow([
                                x_i, y_i, z_i, 
                                z[0].item(), 
                                z[1].item(), 
                                z[2].item(), 
                            ])
        print()

if __name__ == "__main__":
    main()
