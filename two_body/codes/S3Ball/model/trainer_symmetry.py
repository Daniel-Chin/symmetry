import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
from normal_rnn import Conv2dGruConv2d, BATCH_SIZE, LAST_CN_NUM, LAST_H, \
    LAST_W
from ball_data_loader import BallDataLoader
from symmetry import make_translation_batch, make_rotation_batch, do_seq_symmetry, symm_trans, symm_rota
from loss_counter import LossCounter

from common_utils import create_results_path_if_not_exist


def is_need_train(train_config):
    loss_counter = LossCounter([])
    iter_num = loss_counter.load_iter_num(train_config['train_record_path'])
    if train_config['max_iter_num'] > iter_num:
        print("Continue training")
        return True
    else:
        print("No more training is needed")
        return False


def vector_z_score_norm(vector, mean=None, std=None):
    if mean is None:
        mean = torch.mean(vector, [k for k in range(vector.ndim - 1)])
    if std is None:
        std = torch.std(vector, [j for j in range(vector.ndim - 1)])
    return (vector - mean) / std, mean, std


class BallTrainer:
    def __init__(self, config, is_train=True):
        self.model = Conv2dGruConv2d(config)
        self.train_data_loader = BallDataLoader(config['train_data_path'], is_train)
        self.eval_data_loader = BallDataLoader(config['eval_data_path'], not is_train)
        self.mse_loss = nn.MSELoss(reduction='sum')
        device = torch.device('cpu')
        self.model.to(device)
        self.model_path = config['model_path']
        self.kld_loss_scalar = config['kld_loss_scalar']
        self.z_rnn_loss_scalar = config['z_rnn_loss_scalar']
        self.z_symm_loss_scalar = config['z_symm_loss_scalar']
        self.enable_sample = config['enable_sample']
        self.checkpoint_interval = config['checkpoint_interval']
        self.t_batch_multiple = config['t_batch_multiple']
        self.r_batch_multiple = config['r_batch_multiple']
        self.t_range = config['t_range']
        self.r_range = config['r_range']
        self.learning_rate = config['learning_rate']
        self.scheduler_base_num = config['scheduler_base_num']
        self.max_iter_num = config['max_iter_num']
        self.base_len = config['base_len']
        self.train_result_path = config['train_result_path']
        self.train_record_path = config['train_record_path']
        self.eval_record_path = config['eval_record_path']
        self.log_interval = config['log_interval']
        self.eval_interval = config['eval_interval']
        self.sample_prob_param_alpha = config['sample_prob_param_alpha']
        self.sample_prob_param_beta = config['sample_prob_param_beta']
        self.enable_SRS = config['enable_SRS']
        self.is_save_img = config['is_save_img']

    def save_result_imgs(self, img_list, name, seq_len):
        result = torch.cat([img[0] for img in img_list], dim=0)
        save_image(result, self.train_result_path + str(name) + '.png', nrow=seq_len)

    def get_sample_prob(self, step):
        alpha = self.sample_prob_param_alpha
        beta = self.sample_prob_param_beta
        return alpha / (alpha + np.exp((step + beta) / alpha))

    def gen_sample_points(self, base_len, total_len, step, enable_sample):
        if not enable_sample:
            return []
        sample_rate = self.get_sample_prob(step)
        sample_list = []
        for i in range(base_len, total_len):
            r = np.random.rand()
            if r > sample_rate:
                sample_list.append(i)
        return sample_list

    def resume(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(self.model.load_tensor(self.model_path))
            print(f"Model is loaded")
        else:
            print("New model is initialized")

    def batch_encode_to_z(self, x):
        out = self.model.encoder(x)
        mu = self.model.fc11(out.view(out.size(0), -1))
        logvar = self.model.fc12(out.view(out.size(0), -1))
        z1 = self.model.reparameterize(mu, logvar)
        return z1, mu, logvar

    def batch_seq_encode_to_z(self, x: torch.Tensor):
        img_in = x.contiguous().view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        z1, mu, logvar = self.batch_encode_to_z(img_in)
        return [z1.view(x.size(0), x.size(1), z1.size(-1)),
                mu.view(x.size(0), x.size(1), z1.size(-1)),
                logvar.view(x.size(0), x.size(1), z1.size(-1))]

    def batch_decode_from_z(self, z):
        out3 = self.model.fc3(z).view(z.size(0), LAST_CN_NUM, LAST_H, LAST_W)
        frames = self.model.decoder(out3)
        return frames

    def batch_seq_decode_from_z(self, z):
        z_in = z.reshape(z.size(0) * z.size(1), z.size(2))
        recon = self.batch_decode_from_z(z_in)
        return recon.reshape(z.size(0), z.size(1), recon.size(-3), recon.size(-2), recon.size(-1))

    def do_rnn(self, z, hidden):
        out_r, hidden_rz = self.model.rnn(z.unsqueeze(1), hidden)
        z2 = self.model.fc2(out_r.squeeze(1))
        return z2, hidden_rz

    def predict_with_symmetry(self, z_gt, sample_points, symm_func):
        z_SR_seq_batch = []
        hidden_r = torch.zeros([self.model.rnn_num_layers, z_gt.size(0), self.model.rnn_hidden_size])
        for i in range(z_gt.size(1)):
            """Schedule sample"""
            if i in sample_points:
                z_S = z_SR_seq_batch[-1]
            else:
                z = z_gt[:, i]
                z_S = symm_func(z)
            z_SR, hidden_r = self.do_rnn(z_S, hidden_r)
            z_SR_seq_batch.append(z_SR)
        z_x0ESR = torch.stack(z_SR_seq_batch, dim=0).permute(1, 0, 2).contiguous()[:, :-1, :]
        return z_x0ESR

    def eval(self, epoch_num, iter_num, eval_loss_counter):
        print("=====================start eval=======================")
        eval_iter_num = self.eval_data_loader.get_iter_num_of_an_epoch(BATCH_SIZE)
        self.eval_data_loader.set_epoch_num(epoch_num)
        self.model.eval()
        z_gt_list = []
        z0_rnn_list = []
        vae_loss_list = []
        rnn_recon_loss_list = []
        data_shape = None
        for i in range(eval_iter_num):
            data, epoch, progress = self.eval_data_loader.load_a_batch_from_an_epoch(BATCH_SIZE)
            print(progress)
            data = data
            data_shape = data.size()
            z_gt, mu, logvar = self.batch_seq_encode_to_z(data)
            sample_points = list(range(z_gt.size(1)))[self.base_len:]
            z0_rnn = self.predict_with_symmetry(z_gt, sample_points, lambda z: z)
            rnn_recon_loss = self.calc_rnn_loss(data[:, 1:, :, :, :], z_gt, z0_rnn)[0].item()
            vae_loss = self.calc_vae_loss(data, z_gt, mu, logvar)[0].item()
            z_gt_list.append(z_gt.detach())
            z0_rnn_list.append(z0_rnn.detach())
            vae_loss_list.append(vae_loss)
            rnn_recon_loss_list.append(rnn_recon_loss)
        self.model.train()
        tensor_z_gt = torch.stack(z_gt_list)
        tensor_z0_Rnn = torch.stack(z0_rnn_list)
        norm_z_gt, mean_z_gt, std_z_gt = vector_z_score_norm(tensor_z_gt)
        norm_z0_Rnn, mean_z0_Rnn, std_z0_Rnn = vector_z_score_norm(tensor_z0_Rnn, mean_z_gt, std_z_gt)
        rnn_z_loss = nn.MSELoss()(norm_z0_Rnn, norm_z_gt[:, :, 1:, :]).item()
        vae_recon_loss_iter_mean = np.mean(vae_loss_list) / data_shape[1] * (data_shape[1] - 1)
        rnn_recon_loss_iter_mean = np.mean(rnn_recon_loss_list)
        vae_recon_loss_pixel_mean = vae_recon_loss_iter_mean / data_shape[0] / (data_shape[1] - 1) / data_shape[2] / \
                                    data_shape[3] / data_shape[4]
        rnn_recon_loss_pixel_mean = rnn_recon_loss_iter_mean / data_shape[0] / (data_shape[1] - 1) / data_shape[2] / \
                                    data_shape[3] / data_shape[4]
        eval_loss_counter.add_values([vae_recon_loss_iter_mean, rnn_recon_loss_iter_mean,
                                      vae_recon_loss_pixel_mean, rnn_recon_loss_pixel_mean, rnn_z_loss])
        eval_loss_counter.record_and_clear(self.eval_record_path, iter_num, round_idx=4)
        print("=====================end eval=======================")

    def scheduler_func(self, curr_iter):
        return self.scheduler_base_num ** curr_iter

    def train(self):
        create_results_path_if_not_exist(self.train_result_path)
        self.model.train()
        self.resume()
        train_loss_counter = LossCounter(['loss_ED', 'loss_ERnnD', 'loss_Rnn',
                                          'loss_TRnnTr_rnn', 'loss_RRnnRr_rnn',
                                          'loss_TRnnTr_z1', 'loss_RRnnRr_z1',
                                          'KLD'])
        eval_loss_counter = LossCounter(['loss_ED', 'loss_ERnnD', 'loss_ED_mean', 'loss_ERnnD_mean', 'loss_Rnn_norm'])
        iter_num = train_loss_counter.load_iter_num(self.train_record_path)
        epoch_num = self.train_data_loader.get_epoch_num_by_iter_and_batch_size(iter_num, BATCH_SIZE)
        self.train_data_loader.set_epoch_num(epoch_num)
        curr_iter = iter_num
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self.scheduler_func(curr_iter))
        for i in range(iter_num, self.max_iter_num):
            curr_iter = iter_num
            data, new_epoch_num, progress = self.train_data_loader.load_a_batch_from_an_epoch(BATCH_SIZE)
            print(f'{i}, {epoch_num}, {progress * 100}%')
            data = data
            is_log = (i % self.log_interval == 0 and i != 0)
            recon_list = [data[:, 1:, ...]] if is_log and self.is_save_img else None
            is_eval = i % self.eval_interval == 0
            epoch_num = new_epoch_num
            optimizer.zero_grad()
            I_sample_points = self.gen_sample_points(self.base_len, data.size(1), i, self.enable_sample)
            T_sample_points = self.gen_sample_points(self.base_len, data.size(1), i, self.enable_sample)
            R_sample_points = self.gen_sample_points(self.base_len, data.size(1), i, self.enable_sample)
            z_gt, mu, logvar = self.batch_seq_encode_to_z(data)
            T, Tr = make_translation_batch(batch_size=BATCH_SIZE * self.t_batch_multiple, t_range=self.t_range)
            R, Rr = make_rotation_batch(batch_size=BATCH_SIZE * self.r_batch_multiple)
            z0_rnn = self.predict_with_symmetry(z_gt, I_sample_points, lambda z: z)
            vae_loss = self.calc_vae_loss(data, z_gt, mu, logvar, recon_list)
            rnn_loss = self.calc_rnn_loss(data[:, 1:, :, :, :], z_gt, z0_rnn, recon_list)
            T_loss = self.batch_symm_loss(
                z_gt, z0_rnn, T_sample_points, self.t_batch_multiple,
                lambda z: symm_trans(z, T), lambda z: symm_trans(z, Tr)
            )
            R_loss = self.batch_symm_loss(
                z_gt, z0_rnn, R_sample_points, self.r_batch_multiple,
                lambda z: symm_rota(z, R), lambda z: symm_rota(z, Rr)
            )
            loss = self.loss_func(vae_loss, rnn_loss, T_loss, R_loss, train_loss_counter)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if is_log:
                try:
                    self.model.save_tensor(self.model.state_dict(), self.model_path)
                except KeyboardInterrupt:
                    print('Safety qp3948htfaw')
                    self.model.save_tensor(self.model.state_dict(), self.model_path)
                    raise
                print(train_loss_counter.make_record(i))
                train_loss_counter.record_and_clear(self.train_record_path, i)
                self.save_result_imgs(recon_list, f'{i}_{str(I_sample_points)}', z_gt.size(1) - 1)
            if is_eval:
                self.eval(epoch_num - 1, i, eval_loss_counter)
            if i % self.checkpoint_interval == 0 and i != 0:
                self.model.save_tensor(self.model.state_dict(), f'checkpoint_{i}.pt')

    def batch_symm_loss(self, z_gt, z0_rnn, sample_points, symm_batch_multiple, symm_func, symm_reverse_func):
        z_gt_repeat = z_gt.repeat(symm_batch_multiple, 1, 1)
        z0_S_rnn = self.predict_with_symmetry(z_gt_repeat, sample_points, symm_func)
        z0_rnn_repeat = z0_rnn.repeat(symm_batch_multiple, 1, 1)
        zloss_S_rnn_Sr__rnn, zloss_S_rnn_Sr__z1 = \
            self.calc_symm_loss(z_gt_repeat, z0_rnn_repeat, z0_S_rnn, symm_reverse_func)
        return zloss_S_rnn_Sr__rnn / symm_batch_multiple, zloss_S_rnn_Sr__z1 / symm_batch_multiple

    def calc_rnn_loss(self, x1, z_gt, z0_rnn, recon_list=None):
        recon_next = self.batch_seq_decode_from_z(z0_rnn)
        xloss_ERnnD = nn.BCELoss(reduction='sum')(recon_next, x1)
        zloss_Rnn = self.z_rnn_loss_scalar * self.mse_loss(z0_rnn, z_gt[:, 1:, :])
        if recon_list is not None:
            recon_list.append(recon_next.detach())
        return xloss_ERnnD, zloss_Rnn

    def calc_vae_loss(self, data, z_gt, mu, logvar, recon_list=None):
        recon = self.batch_seq_decode_from_z(z_gt)
        recon_loss = nn.BCELoss(reduction='sum')(recon, data)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar)) * self.kld_loss_scalar
        if recon_list is not None:
            recon_list.append(recon[:, 1:].detach())
        return recon_loss, KLD

    def calc_symm_loss(self, z_gt, z0_rnn, z0_S_rnn, symm_reverse_func):
        z0_S_rnn_Sr = do_seq_symmetry(z0_S_rnn, symm_reverse_func)
        z1 = z_gt[:, 1:, :]
        zloss_S_rnn_Sr__rnn = self.z_symm_loss_scalar * self.mse_loss(z0_S_rnn_Sr, z0_rnn)
        zloss_S_rnn_Sr__z1 = self.z_symm_loss_scalar * self.mse_loss(z0_S_rnn_Sr, z1)
        return zloss_S_rnn_Sr__rnn, zloss_S_rnn_Sr__z1

    def loss_func(self, vae_loss, rnn_loss, T_loss, R_loss, loss_counter):
        xloss_ED, KLD = vae_loss
        xloss_ERnnD, zloss_Rnn = rnn_loss
        zloss_T_rnn_Tr__rnn, zloss_T_rnn_Tr__z1 = T_loss
        zloss_R_rnn_Rr__rnn, zloss_R_rnn_Rr__z1 = R_loss

        loss = 0
        loss += xloss_ED + KLD + xloss_ERnnD
        loss += zloss_Rnn
        loss += zloss_T_rnn_Tr__z1 + zloss_R_rnn_Rr__z1
        if self.enable_SRS:
            loss += zloss_T_rnn_Tr__rnn + zloss_R_rnn_Rr__rnn

        loss_counter.add_values([xloss_ED.item(), xloss_ERnnD.item(), zloss_Rnn.item(),
                                 zloss_T_rnn_Tr__rnn.item(), zloss_R_rnn_Rr__rnn.item(),
                                 zloss_T_rnn_Tr__z1.item(), zloss_R_rnn_Rr__z1.item(),
                                 KLD.item()
                                 ])
        return loss

    def eval_a_checkpoint(self, checkpoint_num, checkpoint_path):
        eval_loss_counter = LossCounter(['loss_ED', 'loss_ERnnD', 'loss_ED_mean', 'loss_ERnnD_mean', 'loss_Rnn_norm'])
        self.model.load_state_dict(self.model.load_tensor(checkpoint_path))
        self.eval(0, checkpoint_num, eval_loss_counter)
