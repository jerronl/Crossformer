import numpy as np
import torch
import json


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
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = best_score or np.inf
        self.delta = delta
        self.epoch = 0
        if lradj == "type1":
            self.lr_adjust = {
                2: learning_rate * 0.5**1,
                4: learning_rate * 0.5**2,
                6: learning_rate * 0.5**3,
                8: learning_rate * 0.5**4,
                10: learning_rate * 0.5**5,
            }
        elif lradj == "type2":
            self.lr_adjust = {
                5: learning_rate * 0.5**1,
                10: learning_rate * 0.5**2,
                15: learning_rate * 0.5**3,
                20: learning_rate * 0.5**4,
                25: learning_rate * 0.5**5,
            }
        else:
            self.lr_adjust = {}
        if step > 1:
            self.lr_adjust = {k * step: v for k, v in self.lr_adjust.items()}

    def __call__(self, val_loss, model, path):
        score = val_loss
        if (
            score is None
            or np.isnan(score)
            or self.best_score is not None
            and score > self.best_score + self.delta
        ):
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience} score {score} best {self.best_score}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def adjust_learning_rate(self, optimizer):

        self.epoch += 1
        if self.epoch in self.lr_adjust.keys():
            lr = self.lr_adjust[self.epoch]
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
            (model, -val_loss),
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


import argparse


def init_args():
    parser = argparse.ArgumentParser(description="CrossFormer")

    parser.add_argument("--data", type=str, default="vols", help="data")
    parser.add_argument("--step", type=int, default=1, help="step")
    parser.add_argument("--weight", type=float, default=0.8, help="weight")
    parser.add_argument(
        "--root_path", type=str, default="", help="root path of the data file"
    )
    parser.add_argument("--delta", type=float, default=1.0, help="Huber 损失 δ")
    # parser.add_argument("--thresh",       type=float, default=0.03, help="加权 MSE 阈值")
    # parser.add_argument("--alpha",        type=float, default=1.0, help="极端样本权重 α")
    # parser.add_argument("--tau",          type=float, default=0.9, help="Quantile τ")
    parser.add_argument("--lambda_huber", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=0.5)
    parser.add_argument("--lambda_q90", type=float, default=0.5)
    parser.add_argument("--data_path", type=list, default=".", help="data file")
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

    parser.add_argument("--in_len", type=int, default=20, help="input MTS length (T)")
    parser.add_argument(
        "--out_len", type=int, default=1, help="output MTS length (\tau)"
    )
    parser.add_argument("--seg_len", type=int, default=5, help="segment length (L_seg)")
    parser.add_argument(
        "--win_size", type=int, default=2, help="window size for segment merge"
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=10,
        help="num of routers in Cross-Dimension Stage of TSA (c)",
    )

    parser.add_argument(
        "--d_model", type=int, default=256, help="dimension of hidden states (d_model)"
    )
    parser.add_argument(
        "--d_ff", type=int, default=512, help="dimension of MLP in transformer"
    )
    parser.add_argument("--n_heads", type=int, default=4, help="num of heads")
    parser.add_argument(
        "--e_layers", type=int, default=3, help="num of encoder layers (N)"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout")

    parser.add_argument(
        "--baseline",
        action="store_true",
        help="whether to use mean of past series as baseline for prediction",
        default=False,
    )

    parser.add_argument(
        "--num_workers", type=int, default=0, help="data loader num workers"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
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
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")

    parser.add_argument(
        "--save_pred",
        action="store_true",
        help="whether to save the predicted future MTS",
        default=False,
    )

    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--resume", type=bool, default=True, help="resume")
    parser.add_argument("--query", type=str, default=None, help="resume")
    # parser.add_argument("--use_gpu", type=bool, default=False, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multile gpus"
    )

    args = parser.parse_args(args=[])

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        print(args.gpu)
    return args


def update_args(args, data_parser, itr):
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        for k, v in data_info.items():
            args.__setattr__(k, v)
    if isinstance(args.data_split, str):
        args.data_split = string_split(args.data_split)

    print("Args in experiment:")
    print(args)
    setting = "Crossformer_itr{}_il{}_ol{}_sl{}_win{}_fa{}_dm{}_nh{}_el{}_wt{}".format(
        itr,
        args.in_len,
        args.out_len,
        args.seg_len,
        args.win_size,
        args.factor,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.weight,
    )
    return setting
