from typing import Union, Optional, List, Tuple
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import imageio
import numpy as np
import matplotlib.cbook as mc


BATCH_SIZE = 32
RESULT_PATH = 'Seq_Eval/'
DURATION = 0.2
FIGURE_ROWS = 3


def concatenate_img_seq(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
):
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


def make_one_col_figures(z, recon, z_R, recon_R, col_idx, total_col, title, y_range):
    fig_img_seq = plt.subplot(FIGURE_ROWS, total_col, col_idx + total_col * 0)
    fig_z = plt.subplot(FIGURE_ROWS, total_col, col_idx + total_col * 1)
    fig_z_diff = plt.subplot(FIGURE_ROWS, total_col, col_idx + total_col * 2)
    img_seq = concatenate_img_seq(torch.cat((recon, recon_R), 0), nrow=recon.size(0))
    fig_img_seq.axis('off')
    fig_img_seq.set_title(title)
    fig_img_seq.imshow(img_seq)

    x = np.linspace(0, z.size(0) - 1, z.size(0))
    Z_0 = z[:, 0].cpu().detach().numpy()
    Z_1 = z[:, 1].cpu().detach().numpy()
    Z_2 = z[:, 2].cpu().detach().numpy()
    ZR_0 = z_R[:, 0].cpu().detach().numpy()
    ZR_1 = z_R[:, 1].cpu().detach().numpy()
    ZR_2 = z_R[:, 2].cpu().detach().numpy()

    fig_z.set_xlim(1, x[-1])
    fig_z.set_ylim(y_range[0][0], y_range[0][1])
    fig_z.plot(x, Z_0, color='red', label='z_0')
    fig_z.plot(x, Z_1, color='green', label='z_1')
    fig_z.plot(x, Z_2, color='blue', label='z_2')
    fig_z.plot(x, ZR_0, color='red', linestyle='dashed', label='zR_0')
    fig_z.plot(x, ZR_1, color='green', linestyle='dashed', label='zR_1')
    fig_z.plot(x, ZR_2, color='blue', linestyle='dashed', label='zR_2')
    fig_z.legend()

    Z_0_diff = abs(ZR_0 - Z_0)
    Z_1_diff = abs(ZR_1 - Z_1)
    Z_2_diff = abs(ZR_2 - Z_2)
    fig_z_diff.set_ylim(y_range[1][0], y_range[1][1])
    fig_z_diff.set_xlim(1, x[-1])
    fig_z_diff.plot(x, Z_0_diff, color='red', label='abs(zR_0 - z_0)')
    fig_z_diff.plot(x, Z_1_diff, color='green', label='abs(zR_1 - z_1)')
    fig_z_diff.plot(x, Z_2_diff, color='blue', label='abs(zR_2 - z_2)')
    fig_z_diff.legend()


def make_cols(col_data, titles, y_range):
    total_col = len(col_data)
    for i in range(total_col):
        z, recon, z_R, recon_R = col_data[i]
        make_one_col_figures(
            z,
            recon,
            z_R,
            recon_R,
            i + 1,
            total_col,
            titles[i],
            y_range
        )


def make_gifs(col_data, f_path):
    cols = len(col_data)
    seq_len = col_data[0][0].size(0)
    frame_list = []
    for i in range(seq_len):
        recon_seq = []
        recon_R_seq = []
        for j in range(cols):
            z, recon, z_R, recon_R = col_data[j]
            recon_seq.append(recon[i])
            recon_R_seq.append(recon_R[i])
        tensor = torch.cat((torch.stack(recon_seq, dim=0), torch.stack(recon_R_seq, dim=0)), 0)
        frame = concatenate_img_seq(tensor, nrow=cols)
        frame_list.append(frame)
    imageio.mimsave(f_path, frame_list, 'GIF', duration=DURATION)


def take_one_slice_of_batch_from_data(data, i):
    data_len = len(data)
    data_slice = []
    for j in range(data_len):
        data_slice.append(data[j][i])
    return data_slice


def plot_batch(data_list, title_list, result_path=RESULT_PATH, batch_size=BATCH_SIZE, y_range=((-3, 3), (0, 2))):
    for i in range(batch_size):
        print(i)
        cols = [take_one_slice_of_batch_from_data(data, i) for data in data_list]
        titles = [title[i] for title in title_list]
        plt.figure(figsize=(40, 12))
        make_cols(cols, titles, y_range)
        plt.savefig(f'{result_path}{i}.png')
        # plt.show()
        plt.clf()
        plt.close()
        make_gifs(cols, f'{result_path}{i}.gif')