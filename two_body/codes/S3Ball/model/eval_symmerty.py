from codes.S3Ball.plot_utils import plot_batch
import torch
import numpy as np
import math
from normal_rnn import Conv2dGruConv2d, BATCH_SIZE, LAST_CN_NUM, LAST_H, LAST_W
from codes.S3Ball.ball_data_loader import BallDataLoader
from codes.common_utils import create_results_path_if_not_exist
import codes.S3Ball.symmetry as symm
from train_config import CONFIG


MODEL_PATH = 'Conv2dGruConv2d_symmetry.pt'
DATA_PATH = '../Ball3DImg/32_32_0.2_20_3_init_points_EvalSet_View/'
RESULT_PATH = 'Seq_Eval/'
BASE_LEN = 5


def take_one_slice_of_batch_from_data(data, i):
    data_len = len(data)
    data_slice = []
    for j in range(data_len):
        data_slice.append(data[j][i])
    return data_slice


class SeqEval:
    def __init__(self):
        self.model = Conv2dGruConv2d(CONFIG)
        self.data_loader = BallDataLoader(DATA_PATH)
        self.model.load_state_dict(self.model.load_tensor(MODEL_PATH))
        self.model.eval()

    def do_rnn(self, z, hidden):
        out_r, hidden_rz = self.model.rnn(z.unsqueeze(1), hidden)
        z2 = self.model.fc2(out_r.squeeze(1))
        return z2, hidden_rz

    def batch_encode_to_z(self, x):
        out = self.model.encoder(x)
        mu = self.model.fc11(out.view(out.size(0), -1))
        logvar = self.model.fc12(out.view(out.size(0), -1))
        z1 = self.model.reparameterize(mu, logvar)
        return z1, mu, logvar

    def batch_seq_encode_to_z(self, x):
        img_in = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        z1, mu, logvar = self.batch_encode_to_z(img_in)
        return z1.view(x.size(0), x.size(1), z1.size(-1))

    def batch_decode_from_z(self, z):
        out3 = self.model.fc3(z).view(z.size(0), LAST_CN_NUM, LAST_H, LAST_W)
        frames = self.model.decoder(out3)
        return frames

    def batch_seq_decode_from_z(self, z):
        z_in = z.view(z.size(0) * z.size(1), z.size(2))
        recon = self.batch_decode_from_z(z_in)
        return recon.view(z.size(0), z.size(1), recon.size(-3), recon.size(-2), recon.size(-1))

    def predict_with_symmerty(self, z_gt, base_len, symm_func):
        z_S_batch_seq = symm.do_seq_symmetry(z_gt, symm_func)
        recon_S_batch_seq = self.batch_seq_decode_from_z(z_S_batch_seq)
        recon_SR_seq_batch = []
        z_SR_seq_batch = []
        hidden_r = torch.zeros([self.model.rnn_num_layers, BATCH_SIZE, self.model.rnn_hidden_size])
        for i in range(z_gt.size(1)):
            if i >= base_len:
                z_S = z_SR_seq_batch[-1]
            else:
                z_S = z_S_batch_seq[:, i]
            z_SR, hidden_r = self.do_rnn(z_S, hidden_r)
            z_SR_seq_batch.append(z_SR)
            recon_SR = self.batch_decode_from_z(z_SR)
            recon_SR_seq_batch.append(recon_SR)
        return [
            z_S_batch_seq[:, 1:, :].detach(),
            recon_S_batch_seq[:, 1:, :, :, :].detach(),
            torch.stack(z_SR_seq_batch, dim=0).permute(1, 0, 2).contiguous()[:, :-1, :].detach(),
            torch.stack(recon_SR_seq_batch, dim=0).permute(1, 0, 2, 3, 4).contiguous()[:, :-1, :, :, :].detach()
        ]

    def eval(self):
        create_results_path_if_not_exist(RESULT_PATH)
        data = self.data_loader.load_a_batch_of_random_img_seq(BATCH_SIZE)
        T, Tr = symm.make_translation_batch(batch_size=BATCH_SIZE, is_std_normal=True)
        R, Rr = symm.make_rotation_batch(batch_size=BATCH_SIZE)
        z_gt = self.batch_seq_encode_to_z(data)
        data_gt = self.predict_with_symmerty(z_gt, BASE_LEN, lambda z: z)
        data_gt[1] = data[:, 1:, :, :, :]
        data_symm_trans = self.predict_with_symmerty(z_gt, BASE_LEN, lambda z: symm.symm_trans(z, T))
        data_symm_rota = self.predict_with_symmerty(z_gt, BASE_LEN, lambda z: symm.symm_rota(z, R))
        data_list = [data_gt, data_symm_trans, data_symm_rota]  # col X data_type X batch
        title_list = [
            ["Ground Truth & Prediction" for i in range(BATCH_SIZE)],
            [f'Translation: {np.around(t.cpu().numpy(), 3)}' for t in T],
            [f'Rotation Angle: None' for i in range(BATCH_SIZE)]
        ]
        plot_batch(data_list, title_list, result_path=RESULT_PATH)


if __name__ == "__main__":
    my_eval = SeqEval()
    my_eval.eval()
