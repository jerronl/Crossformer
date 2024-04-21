tables = [
    "volvN.csv",
    "volvT.csv",
    "volvA.csv",
    "volvG.csv",
]
tables = [
    "volvG.csv",
    "volvA.csv",
]
mydrive = "E:/mydoc/git/trade/analyics/"

data_parser = {
    #   "vols": {
    #     "patience":30,
    #     "train_epochs":100,
    #     'data_split':[0.7,0.1,0.2],
    #     'batch_size':512,
    #   },
}

import argparse
import torch

from cross_exp.exp_crossformer import Exp_crossformer
from utils.tools import string_split


parser = argparse.ArgumentParser(description="CrossFormer")

parser.add_argument("--data", type=str, default="vols", help="data")
parser.add_argument(
    "--root_path", type=str, default=mydrive, help="root path of the data file"
)
parser.add_argument("--data_path", type=list, default=tables, help="data file")
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
parser.add_argument("--out_len", type=int, default=1, help="output MTS length (\tau)")
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
parser.add_argument("--e_layers", type=int, default=3, help="num of encoder layers (N)")
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
parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
parser.add_argument(
    "--learning_rate", type=float, default=1e-4, help="optimizer initial learning rate"
)
parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
parser.add_argument("--itr", type=int, default=5, help="experiments times")

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

batch_size = args.batch_size


def update_args(itr):
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        for k, v in data_info.items():
            args.__setattr__(k, v)
    if isinstance(args.data_split, str):
        args.data_split = string_split(args.data_split)

    print("Args in experiment:")
    print(args)
    setting = "Crossformer_il{}_ol{}_sl{}_win{}_fa{}_dm{}_nh{}_el{}_itr{}".format(
        args.in_len,
        args.out_len,
        args.seg_len,
        args.win_size,
        args.factor,
        args.d_model,
        args.n_heads,
        args.e_layers,
        itr,
    )
    return setting


data_parser = {
    "vols": {
        "patience": 1,
        "train_epochs": 1,
        "learning_rate": 2e-3,
        "data_split": [0.7, 0.15, 0.15],
        "batch_size": batch_size * 2 // 5,
        "e_layers": 5,
        "d_model": 512,
        "lradj": "type2",
        "itr": 3,
        # "query": "date>'#2024-02-15' and date<'#2024-03-22'",
    },
}

tables = [
    "volvN.csv",
]
data_parser = {
    "vols": {
        "e_layers": 5,
        "d_model": 512,
        "query": "date>'#2024-02-30' and date<'#2024-04-22' "#and e2d_20==3",
    },
}
from data.data_loader import DatasetMTS
import pandas as pd

results = []
DatasetMTS.clear()
setting = update_args(0)
exp = Exp_crossformer(args)
# df=pd.read_csv(r'E:\mydoc\git\trade/input.csv')
# results.append(
#     exp.test(
#         data="vols",
#         save_pred=True,
#         inverse=True,
#         run_metric=False,
#         data_path=df,
#     )
# )
print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
for table in tables:
    results.append(
        exp.test(
            data="vols",
            save_pred=True,
            inverse=True,
            run_metric=False,
            data_path=[table],
        )
    )
    results.append(
        exp.test(
            data="prcs",
            save_pred=True,
            inverse=True,
            run_metric=False,
            data_path=[table],
        )
    )
# import numpy as np
# for i in range(2):
#     for j in range(2):
#         np.savetxt(f'./res{i}{j}.csv',results[i][j],delimiter=",")
for ii in range(args.itr):
    # setting record of experiments
    setting = update_args(ii)

    exp = Exp_crossformer(args)  # set experiments
    print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
    tables = ["volvG.csv", "volvN.csv"]
    exp.train(setting, "vols")

    print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    preds, trues = exp.test(setting, "vols", True, data_path=[tables[0]], inverse=True)
    print(preds.shape, trues.shape)

    exp.train(setting, "prcs")
    for table in tables:
        preds, trues = exp.test(setting, "prcs", True, data_path=[table], inverse=True)
        print(preds.shape, trues.shape)
