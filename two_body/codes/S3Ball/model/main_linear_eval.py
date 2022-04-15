from train_config import CONFIG
from linear_eval import create_path_and_eval_checkpoint

EVAL_CHECK_POINT_PATH = "Conv2dGruConv2d_symmetry.pt"
create_path_and_eval_checkpoint(EVAL_CHECK_POINT_PATH, f'LINEAR_{EVAL_CHECK_POINT_PATH.split(".")[0]}/', CONFIG)