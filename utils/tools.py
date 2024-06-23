import numpy as np
import torch
import json


class EarlyStopping:
    def __init__(
        self, lradj, learning_rate, patience=7, verbose=False, delta=0, best_score=None
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
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

    def __call__(self, val_loss, model, path):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score > self.best_score + self.delta:
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
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...",
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
