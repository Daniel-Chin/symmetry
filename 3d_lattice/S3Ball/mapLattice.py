from model.normal_rnn import Conv2dGruConv2d, LAST_CN_NUM, LAST_H, LAST_W, IMG_CHANNEL
from model.train_config import CONFIG
import os
import torch
from PIL import Image

from shared import *

CHECKPOINTS_PATH = './model/checkpoint_%d.pt'
CHECKPOINT_INTERVAL = 10000

device = torch.device("cpu")

def loadNNs():
    models = []
    i = 0
    while True:
        i += CHECKPOINT_INTERVAL
        checkpoint_path = CHECKPOINTS_PATH % i
        if os.path.exists(checkpoint_path):
            model = Conv2dGruConv2d(CONFIG).to(device)
            model.eval()
            model.load_state_dict(torch.load(
                checkpoint_path, 
                map_location=device, 
            ))
            models.append(model)
        else:
            return models

def loadDataset():
    prev_cwd = os.getcwd()
    os.chdir(DATASET_PATH)
    dataset = torch.zeros((
        X_LATTICE_LEN, 
        Y_LATTICE_LEN, 
        Z_LATTICE_LEN, 
        3, 
        IMG_W, 
        IMG_H, 
    ))
    for x_i in range(X_LATTICE_LEN):
        for y_i in range(Y_LATTICE_LEN):
            for z_i in range(Z_LATTICE_LEN):
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
    nns = loadNNs()
    dataset = loadDataset()

if __name__ == "__main__":
    main()
