tables = [
    "volvN.csv",
    "volvT.csv",
    "volvA.csv",
    "volvG.csv",
]
# tables = [
#     "volvG.csv",
#     "volvA.csv",
# ]
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

# tables = [
#     "volvN.csv",
# ]
# data_parser = {
#     "vols": {
#         "e_layers": 5,
#         "d_model": 512,
#         "query": "date>'#2024-02-30' and date<'#2024-04-22' "#and e2d_20==3",
#     },
# }
from data.data_loader import DatasetMTS
import pandas as pd

import seaborn as sns, numpy as np, math
import matplotlib.pyplot as plt


def regplot(cols, figsize=(16, 16)):
    global metrics
    cnt = len(dep_var) - dep_var.count("_")
    figs = min(cnt, cols)
    _, axes = plt.subplots(math.ceil(cnt / figs), figs, figsize=figsize)
    j = 0

    for i, name in enumerate(dep_var):
        if name != "_":
            axs = axes.flat[j] if figs > 1 else axes
            j = j + 1
            left, right = 999, -999
            for ii in range(len(results)):
                preds, trues, _ = results[ii]
                sns.regplot(
                    ax=axs,
                    x=trues[:, i],
                    y=preds[:, i],
                    scatter_kws={"color": f"C{ii}", "alpha": 0.3},
                    line_kws={"color": f"C{ii}", "alpha": 0.3},
                    label=labels[1][ii],
                )
                mask = ~np.isnan(trues[:, i])
                if not dep_var[i][:3] in ["dtm", "pmc"]:
                    left = min(left, max(np.min(trues[:, i][mask]), -5))
                    right = max(right, min(np.max(trues[:, i][mask]), 5))
                else:
                    left = min(left, np.min(trues[:, i][mask]))
                    right = max(right, np.max(trues[:, i][mask]))
            axs.set_title(name)
            axs.set_xlim(left=left, right=right)
            axs.legend()
    metric = []
    for ii in range(len(results)):
        _, _, m = results[ii]
        metric.append(m)

    metrics = np.append(
        metrics, np.array(metric).reshape([1, len(metric), len(m)]), axis=0
    )

    plt.show()


def plot_metric(*args, **kwargs):
    a, b, c = metrics.shape
    _, axs = plt.subplots(
        nrows=math.ceil(c / 2),
        ncols=2,
        figsize=(16, 16),
    )
    for i in range(c):
        ax = (
            axs[i // 2, i % 2] if c > 1 else axs
        )  # Handle the case when c=1 to avoid indexing errors
        for j in range(a):
            ax.plot(
                metrics[j, :, i], label=labels[0][j], *args, **kwargs
            )  # Plot each series in the i-th plot
        ax.set_title(labels[2][i])
        ax.legend()  # Show legend in each subplot

        # Set custom x-axis labels
        ax.set_xticks(range(b))  # Set x-tick positions for all 'b' points
        ax.set_xticklabels(labels[1])  # Set x-tick labels

    plt.tight_layout()
    plt.show()


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
cutdate = "2024-04-30"
metrics_cnt = 6
itr = 5
labels = [
    [f"m{i}" for i in range(itr)],
    [f"h{h+1}" for h in range(5)],
    ["mae", "mse", "rmse", "mape", "mspe", "accr"],
]
dep_var = [
    "level0",
    "slope0",
    "curve0",
    "level1",
    "slope1",
    "curve1",
    "level2",
    "slope2",
    "curve2",
    "level3",
    "slope3",
    "curve3",
]
metrics = np.empty((0, len(labels[1]), len(labels[2])))
for i in range(itr):
    results = []
    for h in range(5):
        data_parser = {
            "vols": {
                "e_layers": 5,
                "d_model": 512,
                "lradj": "type2",
                "query": f"date>'#{cutdate}' and floor(horizon)=={h+1} and e2d_20==17",
            },
        }
        setting = update_args(i)
        DatasetMTS.clear()
        exp = Exp_crossformer(args)
        print(
            f">>>>>>>testing : {data_parser['vols']['query']} m{i}h{h+1}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        )
        results.append(exp.test(setting, "vols", True, inverse=True))
    regplot(3)

data_parser = {
    "vols": {
        "e_layers": 5,
        "d_model": 512,
        "query": f"date>'#{cutdate}' and horizon==1",
    },
}
for i in range(args.itr):
    setting = update_args(i)
    DatasetMTS.clear()
    exp = Exp_crossformer(args)
    print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    results = []
    for table in tables:
        results.append(exp.test(setting, "vols", True, data_path=[table], inverse=True))
#   regplot(3)

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
tables = [
    "volvG.csv",
    # "volvA.csv",
]
data_parser = {
    "vols": {
        "e_layers": 5,
        "d_model": 512,
        "patience": 3,
        "train_epochs": 5,
        "data_path": tables,
    },
}
for ii in range(args.itr):
    # setting record of experiments
    setting = update_args(ii)

    exp = Exp_crossformer(args)  # set experiments
    print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
    tables = ["volvT.csv", "volvN.csv"]
    exp.train(setting, "vols")

    print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    preds, trues = exp.test(setting, "vols", True, data_path=[tables[0]], inverse=True)
    print(preds.shape, trues.shape)

    exp.train(setting, "prcs")
    for table in tables:
        preds, trues = exp.test(setting, "prcs", True, data_path=[table], inverse=True)
        print(preds.shape, trues.shape)

dep_var = [
    "pmcat",
    "close",
    "hi",
    "lo",
]
for i in range(args.itr):
    setting = update_args(i)
    DatasetMTS.clear()
    exp = Exp_crossformer(args)
    print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    results = []
    for table in tables:
        results.append(exp.test(setting, "prcs", True, data_path=[table], inverse=True))
