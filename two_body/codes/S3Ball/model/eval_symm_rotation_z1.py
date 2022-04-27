import math

from eval_symmerty import SeqEval, BASE_LEN
from codes.common_utils import create_results_path_if_not_exist, make_a_ndarray_batch
from codes.S3Ball.symmetry import rota_Y
import numpy as np
import codes.S3Ball.symmetry as symm
from codes.S3Ball.plot_utils import plot_batch
import math


RESULT_PATH = 'eval_symm_rotation_z1/'
BATCH_SIZE = 32


if __name__ == "__main__":
    evaler = SeqEval()
    create_results_path_if_not_exist(RESULT_PATH)
    data = evaler.data_loader.load_a_batch_of_fixed_img_seq(BATCH_SIZE)
    z_gt = evaler.batch_seq_encode_to_z(data)
    data_gt = evaler.predict_with_symmerty(z_gt, BASE_LEN, lambda z: z)
    gt_data = [
        data_gt[0],
        data[:, 1:, :, :, :],
        data_gt[2],
        data_gt[3],
    ]

    a1 = -math.pi / 6
    a2 = -math.pi / 12
    a3 = math.pi / 12
    a4 = math.pi / 6
    r1 = make_a_ndarray_batch(rota_Y(a1)[0], BATCH_SIZE)
    r2 = make_a_ndarray_batch(rota_Y(a2)[0], BATCH_SIZE)
    r3 = make_a_ndarray_batch(rota_Y(a3)[0], BATCH_SIZE)
    r4 = make_a_ndarray_batch(rota_Y(a4)[0], BATCH_SIZE)

    data_symm_trans_1 = evaler.predict_with_symmerty(z_gt, BASE_LEN, lambda z: symm.symm_rota(z, r1))
    data_symm_trans_2 = evaler.predict_with_symmerty(z_gt, BASE_LEN, lambda z: symm.symm_rota(z, r2))
    data_symm_trans_3 = evaler.predict_with_symmerty(z_gt, BASE_LEN, lambda z: symm.symm_rota(z, r3))
    data_symm_trans_4 = evaler.predict_with_symmerty(z_gt, BASE_LEN, lambda z: symm.symm_rota(z, r4))

    data_list = [data_symm_trans_1, data_symm_trans_2, gt_data, data_symm_trans_3, data_symm_trans_4]
    title_list = [
        [f'Rotation Angle: {round(a1 / math.pi * 180, 3)}°' for i in range(BATCH_SIZE)],
        [f'Rotation Angle: {round(a2 / math.pi * 180, 3)}°' for i in range(BATCH_SIZE)],
        ["Ground Truth & Prediction" for i in range(BATCH_SIZE)],
        [f'Rotation Angle: {round(a3 / math.pi * 180, 3)}°' for i in range(BATCH_SIZE)],
        [f'Rotation Angle: {round(a4 / math.pi * 180, 3)}°' for i in range(BATCH_SIZE)]
    ]
    plot_batch(data_list, title_list, result_path=RESULT_PATH)

