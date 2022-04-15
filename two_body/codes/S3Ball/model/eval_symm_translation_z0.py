from eval_symmerty import SeqEval, BASE_LEN
from codes.common_utils import create_results_path_if_not_exist, make_a_ndarray_batch
from codes.S3Ball.symmetry import rota_Y
import numpy as np
import codes.S3Ball.symmetry as symm
from codes.S3Ball.plot_utils import plot_batch


RESULT_PATH = 'eval_symm_translation_z0/'
BATCH_SIZE = 32


if __name__ == "__main__":
    evaler = SeqEval()
    create_results_path_if_not_exist(RESULT_PATH)
    data = evaler.data_loader.load_a_batch_of_fixed_img_seq(BATCH_SIZE)
    z_gt = evaler.batch_seq_encode_to_z(data)
    data_gt = evaler.predict_with_symmerty(z_gt, BASE_LEN, lambda z: z)
    # data_gt[1] = data[:, 1:, :, :, :]
    gt_data = [
        data_gt[0],
        data[:, 1:, :, :, :],
        data_gt[2],
        data_gt[3],
    ]

    t1 = make_a_ndarray_batch(np.array([-0.2, 0.0, 0.0]), BATCH_SIZE)
    t2 = make_a_ndarray_batch(np.array([-0.1, 0.0, 0.0]), BATCH_SIZE)
    t3 = make_a_ndarray_batch(np.array([0.1, 0.0, 0.0]), BATCH_SIZE)
    t4 = make_a_ndarray_batch(np.array([0.2, 0.0, 0.0]), BATCH_SIZE)

    data_symm_trans_1 = evaler.predict_with_symmerty(z_gt, BASE_LEN, lambda z: symm.symm_trans(z, t1))
    data_symm_trans_2 = evaler.predict_with_symmerty(z_gt, BASE_LEN, lambda z: symm.symm_trans(z, t2))
    data_symm_trans_3 = evaler.predict_with_symmerty(z_gt, BASE_LEN, lambda z: symm.symm_trans(z, t3))
    data_symm_trans_4 = evaler.predict_with_symmerty(z_gt, BASE_LEN, lambda z: symm.symm_trans(z, t4))

    data_list = [data_symm_trans_1, data_symm_trans_2, gt_data, data_symm_trans_3, data_symm_trans_4]
    title_list = [
        [f'Translation: {np.around(t.cpu().numpy(), 3)}' for t in t1],
        [f'Translation: {np.around(t.cpu().numpy(), 3)}' for t in t2],
        ["Ground Truth & Prediction" for i in range(BATCH_SIZE)],
        [f'Translation: {np.around(t.cpu().numpy(), 3)}' for t in t3],
        [f'Translation: {np.around(t.cpu().numpy(), 3)}' for t in t4]
    ]
    plot_batch(data_list, title_list, result_path=RESULT_PATH)

