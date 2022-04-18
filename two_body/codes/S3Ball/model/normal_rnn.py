import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import gzip
from symmetry import make_rotation_Y_batch, make_translation_batch

# todo: make these parameters configurable
BATCH_SIZE = 32
log_interval = 10
IMG_CHANNEL = 3

LAST_H = 4
LAST_W = 4

FIRST_CH_NUM = 64
LAST_CN_NUM = FIRST_CH_NUM * 4

RNN_INPUT_SIZE = 6
RNN_OUT_FEATURES = 6


class Conv2dGruConv2d(nn.Module):
    def __init__(self, config):
        super(Conv2dGruConv2d, self).__init__()
        self.rnn_hidden_size = config['rnn_hidden_size']
        self.rnn_num_layers = config['rnn_num_layers']
        self.latent_code_num = config['latent_code_num']

        self.encoder = nn.Sequential(
            nn.Conv2d(IMG_CHANNEL, FIRST_CH_NUM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(FIRST_CH_NUM, FIRST_CH_NUM * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(FIRST_CH_NUM * 2, LAST_CN_NUM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc11 = nn.Linear(LAST_CN_NUM * LAST_H * LAST_W, self.latent_code_num)
        self.fc12 = nn.Linear(LAST_CN_NUM * LAST_H * LAST_W, self.latent_code_num)

        self.rnn = nn.RNN(
            input_size=RNN_INPUT_SIZE,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True,
        )
        self.fc2 = nn.Linear(in_features=self.rnn_hidden_size, out_features=RNN_OUT_FEATURES)

        self.fc3 = nn.Sequential(
            nn.Linear(self.latent_code_num, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, LAST_CN_NUM * LAST_H * LAST_W)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(LAST_CN_NUM, FIRST_CH_NUM * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(FIRST_CH_NUM * 2, FIRST_CH_NUM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(FIRST_CH_NUM, IMG_CHANNEL, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def save_tensor(self, tensor, path):
        """保存 tensor 对象到文件"""
        torch.save(tensor, path)

    def load_tensor(self, path):
        """从文件读取 tensor 对象"""
        return torch.load(path)

    def reparameterize(self, mu, logvar):
        eps = Variable(torch.randn(mu.size(0), mu.size(1)))
        z = mu + eps * torch.exp(logvar) * 0.5
        return z
