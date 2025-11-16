import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(((pred - true) / true) ** 2)


def fuzzy_accuracy(pred_logits, true_labels):
    pred_class = np.argmax(pred_logits, axis=-1)
    diff = np.abs(pred_class - true_labels)
    score = np.where(diff == 0, 1.0, np.where(diff == 1, 0.25, 0.0))
    return score.mean()


def mean_bucket_distance(pred_logits, true_labels):
    pred_class = np.argmax(pred_logits, axis=-1)
    diff = np.abs(pred_class - true_labels)
    return diff.mean()


def make_metric(ycat):
    def metric_cat(pred, true):
        tv = to_numpy(true)
        tc = to_numpy(true[:,0])

        if isinstance(pred, np.ndarray):
            iv = pred[:, :-ycat]
            ic = pred[:, -ycat:]
        else:
            p = pred.detach().cpu().numpy()
            iv = p[:, :, :-ycat]
            ic = p[:, 0, -ycat:]

        iv = np.asarray(iv)
        ic = np.asarray(ic)

        mae = MAE(iv, tv)
        mse = MSE(iv, tv)
        rmse = RMSE(iv, tv)
        mape = MAPE(iv, tv)
        mbd = mean_bucket_distance(ic, tc)
        accr = fuzzy_accuracy(ic, tc)
        return mae, mse, rmse, mape, mbd, accr

    def metric(pred, true):
        if isinstance(true, (tuple, list)):
            tv_raw = true[0]
        else:
            tv_raw = true

        tv = to_numpy(tv_raw)
        iv = to_numpy(pred)
        mae = MAE(iv, tv)
        mse = MSE(iv, tv)
        rmse = RMSE(iv, tv)
        mape = MAPE(iv, tv)
        mspe = MSPE(iv, tv)
        return mae, mse, rmse, mape, mspe, -1.0

    return metric_cat if ycat > 0 else metric
