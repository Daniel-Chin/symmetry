import torch
import torch.nn as nn
import torch.optim as optim
import os
from codes.S3Ball.ball_data_loader import BallDataLoader
from codes.common_utils import create_results_path_if_not_exist
from normal_rnn import Conv2dGruConv2d, BATCH_SIZE, LAST_CN_NUM, LAST_H, LAST_W
from codes.loss_counter import LossCounter


LINEAR_EVAL_BATCH_SIZE = 32
LINEAR_EVAL_DATA_PATH = '../Ball3DImg/32_32_0.2_20_3_init_points_EvalSet/'

Z_LEN = 3
EPOACH = 25001
INTERVAL = 50


def load_params_if_exit(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Params loaded from {path}")
        print(f'linear_whole_params:\n {model.weight}\n{model.bias}\n')
    else:
        print(f'New params created')


class LinearEval:
    def __init__(self, checkpoint, result_path, config):
        self.model = Conv2dGruConv2d(config)
        self.data_loader = BallDataLoader(LINEAR_EVAL_DATA_PATH, True)
        self.model.load_state_dict(self.model.load_tensor(checkpoint))
        self.model.eval()
        self.linear_whole = nn.Linear(Z_LEN, Z_LEN)
        self.linear_xz = nn.Linear(2, 2)
        self.linear_record_path = f'{result_path}record.txt'
        self.linear_whole_model_path = f'{result_path}linear_whole.pt'
        self.linear_xz_model_path = f'{result_path}linear_xz.pt'
        self.params_path = f'{result_path}Params.txt'

    def get_data(self):
        img, position = self.data_loader.load_a_batch_of_random_img_seq_with_position(LINEAR_EVAL_BATCH_SIZE)
        """eliminate the first one"""
        img = img[:, 1:, :, :, :]
        img = img.contiguous().view(img.size(0) * img.size(1), img.size(2), img.size(3), img.size(4))
        position = position[:, 1:, :]
        position = position.contiguous().view(position.size(0) * position.size(1), position.size(2))
        position_xz = position[:, ::2]
        return img, position, position_xz

    def encode_img_to_z(self, x):
        out = self.model.encoder(x)
        z = self.model.fc11(out.view(out.size(0), -1))
        return z

    def train_linear(self):
        loss_counter = LossCounter(["loss_whole", "loss_xz"])

        load_params_if_exit(self.linear_whole, self.linear_whole_model_path)
        load_params_if_exit(self.linear_xz, self.linear_xz_model_path)
        optimizer_whole = optim.Adam(self.linear_whole.parameters())
        optimizer_xz = optim.Adam(self.linear_xz.parameters())

        for i in range(loss_counter.load_iter_num(self.linear_record_path), EPOACH):
            is_log = i != 0 and i % INTERVAL == 0
            optimizer_whole.zero_grad()
            optimizer_xz.zero_grad()
            img, position, position_xz = self.get_data()

            z_whole = self.encode_img_to_z(img).detach()
            z_xz = z_whole[:, ::2]

            pred_whole = self.linear_whole(z_whole)
            pred_xz = self.linear_xz(z_xz)
            loss_whole = nn.MSELoss()(pred_whole, position)
            loss_xz = nn.MSELoss()(pred_xz, position_xz)

            loss_counter.add_values([loss_whole.item(), loss_xz.item()])

            loss_whole.backward()
            loss_xz.backward()

            optimizer_whole.step()
            optimizer_xz.step()

            if is_log:
                print(loss_counter.make_record(i))
                loss_counter.record_and_clear(self.linear_record_path, i)
                torch.save(self.linear_whole.state_dict(), self.linear_whole_model_path)
                torch.save(self.linear_xz.state_dict(), self.linear_xz_model_path)
                linear_whole_str = f'linear_whole_params:\n {self.linear_whole.weight}\n{self.linear_whole.bias}\n'
                linear_xz_str = f'linear_xz_params:\n {self.linear_xz.weight}\n{self.linear_xz.bias}\n'
                print(linear_whole_str)
                print(linear_xz_str)
                fo = open(self.params_path, 'w')
                fo.writelines(linear_whole_str)
                fo.writelines(linear_xz_str)
                fo.close()


def create_path_and_eval_checkpoint(checkpoint, result_path, config):
    create_results_path_if_not_exist(result_path)
    linear_eval = LinearEval(checkpoint, result_path, config)
    linear_eval.train_linear()
