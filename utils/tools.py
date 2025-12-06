import os
import json
import argparse
import numpy as np
import torch


class EarlyStopping:
    def __init__(
        self,
        lradj,
        learning_rate,
        patience=7,
        verbose=False,
        delta=0,
        best_score=None,
        step=1,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score or np.inf
        self.early_stop = False
        self.val_loss_min = best_score or np.inf
        self.delta = delta
        self.step = step
        self.steps = 0
        self.stage = 0
        self.stages = 4
        self.threshold = {"type1": 2, "type2": 5}[lradj]
        self.lr_adjust = [learning_rate * 0.5**i for i in range(self.stages + 1)]
        self.prev_val = float("inf")

    def __call__(self, val_loss, model, path):
        score = val_loss
        self.adj = False
        if (
            score is None
            or np.isnan(score)
            or self.best_score is not None
            and score > self.best_score + self.delta
        ):
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience} score {score:.4g} best {self.best_score:.4g}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
            else:
                if self.stage < self.stages and val_loss > self.prev_val:
                    self.steps += self.step
                    if self.steps >= self.threshold:
                        self.steps = 0
                        self.stage += 1
                        self.adj = True
                        self.prev_val = float("inf")
                else:
                    self.prev_val = val_loss
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            self.steps = 0
            return True

    def adjust_learning_rate(self, optimizer):
        if self.adj:
            lr = self.lr_adjust[self.stage]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            print_color(94, f"Updating learning rate to {lr}")
            return True
        return False

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print_color(
                92,
                f"Validation loss decreased ({self.val_loss_min:.4g} --> {val_loss:.4g}).  Saving model ...",
            )
        torch.save(
            model,
            path + "/" + "checkpoint.pth",
        )
        self.val_loss_min = val_loss


def load_args(filename):
    with open(filename, "r") as f:
        args = json.load(f)
    return args


def string_split(str_for_split):
    str_no_space = str_for_split.replace(" ", "")
    str_split = str_no_space.split(",")
    value_list = [eval(x) for x in str_split]

    return value_list


def print_color(color, *args, **kwargs):
    print(*((f"\033[{color}m",) + args + ("\033[0m",)), **kwargs)


_DEFAULT_ARGS_DICT = None


def init_args():
    global _DEFAULT_ARGS_DICT
    parser = argparse.ArgumentParser(description="CrossFormer")

    parser.add_argument("--data", type=str, default="vols", help="data")
    parser.add_argument("--step", type=int, default=1, help="step")
    parser.add_argument("--weight", type=float, default=0.8, help="weight")
    parser.add_argument(
        "--root_path", type=str, default="", help="root path of the data file"
    )
    parser.add_argument(
        "--data_path",
        type=list,
        default=".",
        help="data file",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="0.7,0.1,0.2",
        help="train/val/test split, can be ratio or number",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location to store model checkpoints",
    )

    parser.add_argument(
        "--weight_over",
        type=float,
        default=1.5,
        help="extra weight for over-estimation in MSE",
    )
    parser.add_argument(
        "--weight_under",
        type=float,
        default=1.0,
        help="extra weight for under-estimation in MSE",
    )
    parser.add_argument(
        "--gain_reg",
        type=float,
        default=1.0,
        help="gain factor for regression error",
    )
    parser.add_argument(
        "--tail_k",
        type=float,
        default=3.0,
        help="threshold multiplier for tail loss",
    )
    parser.add_argument(
        "--lambda_tail",
        type=float,
        default=0.0,
        help="weight for tail loss",
    )
    parser.add_argument(
        "--lambda_anti",
        type=float,
        default=0.0,
        help="weight for anti-shrinkage loss",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.5,
        help="quantile for quantile loss",
    )
    parser.add_argument(
        "--lambda_quantile",
        type=float,
        default=0.0,
        help="weight for quantile loss",
    )

    parser.add_argument(
        "--dist_alpha",
        type=float,
        default=3.0,
        help="distance decay factor for soft label (larger -> more peaked)",
    )
    parser.add_argument(
        "--lambda_ce",
        type=float,
        default=1.0,
        help="weight for classification CE part",
    )
    parser.add_argument(
        "--use_expected_dist_penalty",
        type=bool,
        default=False,
        help="whether to use expected distance penalty",
    )
    parser.add_argument(
        "--lambda_dist",
        type=float,
        default=0.0,
        help="weight for expected distance penalty",
    )
    parser.add_argument(
        "--lambda_mse",
        type=float,
        default=1.0,
        help="weight for mse regression loss",
    )
    parser.add_argument(
        "--use_var_loss",
        type=bool,
        default=True,
        help="whether to use variance regularization loss",
    )
    parser.add_argument(
        "--lambda_var",
        type=float,
        default=0.5,
        help="weight for variance loss",
    )
    parser.add_argument(
        "--lambda_var_beta",
        type=float,
        default=1.0,
        help="exponent for variance loss (beta)",
    )

    parser.add_argument("--in_len", type=int, default=20, help="input length of series")
    parser.add_argument(
        "--out_len", type=int, default=1, help="output length of series"
    )
    parser.add_argument(
        "--seg_len", type=int, default=5, help="segment length for Crossformer"
    )
    parser.add_argument(
        "--win_size", type=int, default=2, help="window size for Crossformer"
    )

    parser.add_argument("--factor", type=int, default=10, help="factor for Crossformer")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--d_ff", type=int, default=512, help="dimension of fcn")
    parser.add_argument("--n_heads", type=int, default=4, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=5, help="num of enc layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout")
    parser.add_argument(
        "--baseline",
        type=bool,
        default=False,
        help="whether to use baseline implementation",
    )

    parser.add_argument(
        "--num_workers", type=int, default=0, help="data loader num workers"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="batch size of train input data"
    )

    parser.add_argument("--train_epochs", type=int, default=20, help="train epochs")
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="optimizer initial learning rate",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="optimizer type",
    )
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")

    parser.add_argument(
        "--save_pred",
        action="store_true",
        help="whether to save the predicted future MTS",
        default=False,
    )

    parser.add_argument("--profile_mode", type=bool, default=False, help="profile mode")
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--resume", type=bool, default=True, help="resume from ckpt")
    parser.add_argument("--query", type=str, default=None, help="query filter for data")

    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multiple gpus"
    )

    args = parser.parse_args(args=[])
    _DEFAULT_ARGS_DICT = args.__dict__.copy()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    return args


def format_nested(data, fmt=".4g"):
    if isinstance(data, (list, tuple)):
        open_bracket = "(" if isinstance(data, tuple) else "["
        close_bracket = ")" if isinstance(data, tuple) else "]"
        return (
            open_bracket
            + ", ".join(format_nested(item, fmt) for item in data)
            + close_bracket
        )
    elif isinstance(data, np.ndarray) and data.ndim > 0:
        return "[" + ", ".join(format_nested(item, fmt) for item in data) + "]"
    else:
        return f"{data:{fmt}}"


def string_split(str_for_split):
    str_no_space = str_for_split.replace(" ", "")
    str_split = str_no_space.split(",")
    value_list = [eval(x) for x in str_split]
    return value_list


def update_args(args, data_parser, itr, arg_set="vols"):
    data_info = data_parser.get(arg_set, {}).copy()
    data_info["data"] = arg_set

    for k, v in data_info.items():
        setattr(args, k, v)

    if isinstance(args.data_split, str):
        args.data_split = string_split(args.data_split)

    if _DEFAULT_ARGS_DICT is None:
        init_args()

    default_args = _DEFAULT_ARGS_DICT

    SWEEP_KEYS = []

    EXCLUDE_KEYS = [
        "data_split",
        "checkpoints",
        "data_path",
        "devices",
        "gpu",
        "use_multi_gpu",
        "root_path",
        "optim",
        "lradj",
        "num_workers",
        "save_pred",
        "profile_mode",
        "resume",
        "query",
        "data",
        "use_gpu",
        "itr",
        "batch_size",
        "learning_rate",
        "patience",
        "train_epochs",
    ]

    for k, current_v in args.__dict__.items():
        if k in EXCLUDE_KEYS:
            continue

        if k in default_args:
            default_v = default_args[k]

            if current_v is None and default_v is None:
                continue

            if isinstance(default_v, (list, tuple)):
                if str(current_v) != str(default_v):
                    SWEEP_KEYS.append(k)
            elif current_v != default_v:
                SWEEP_KEYS.append(k)
        elif current_v != 0:
            SWEEP_KEYS.append(k)

    sorted_sweep_keys = sorted(SWEEP_KEYS)
    sweep_part = "_".join(f"{k}{getattr(args, k)}" for k in sorted_sweep_keys)

    base_setting = "Cf_{data}_itr{itr}"

    if sweep_part:
        setting_template = base_setting + "_sw_" + sweep_part
    else:
        setting_template = base_setting

    setting = setting_template.format(
        itr=args.itr,
        data=args.data,
    )

    return setting
