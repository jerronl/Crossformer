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


def softmax(x, axis=-1):
    x = np.asarray(x)
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def mean_bucket_distance(pred_logits, true_labels):
    probs = softmax(pred_logits, axis=-1)
    class_idx = np.arange(probs.shape[-1], dtype=np.float64)
    expected = np.sum(probs * class_idx, axis=-1)
    true_labels = true_labels.astype(np.float64)
    diff = np.abs(expected - true_labels)
    return diff.mean()


def balanced_mean_bucket_distance(pred_logits, true_labels, num_classes=None):
    probs = softmax(pred_logits, axis=-1)
    class_idx = np.arange(probs.shape[-1], dtype=np.float64)
    expected = np.sum(probs * class_idx, axis=-1)
    true_labels = true_labels.astype(np.float64)
    if num_classes is None:
        num_classes = int(true_labels.max()) + 1
    dists = []
    for k in range(num_classes):
        mask = true_labels == k
        if not np.any(mask):
            continue
        d_k = np.abs(expected[mask] - float(k)).mean()
        dists.append(d_k)
    if not dists:
        return 0.0
    return float(np.mean(dists))


def balanced_within_one_accuracy(pred_logits, true_labels, num_classes=None):
    probs = softmax(pred_logits, axis=-1)
    class_idx = np.arange(probs.shape[-1], dtype=np.float64)
    expected = np.sum(probs * class_idx, axis=-1)
    true_labels = true_labels.astype(np.float64)
    diff = np.abs(expected - true_labels)
    if num_classes is None:
        num_classes = int(true_labels.max()) + 1
    acc_per_class = []
    for k in range(num_classes):
        mask = true_labels == k
        if not np.any(mask):
            continue
        acc_k = (diff[mask] <= 1.0).mean()
        acc_per_class.append(acc_k)
    if not acc_per_class:
        return 0.0
    return float(np.mean(acc_per_class))


def make_metric(ycat):
    def metric_cat(pred, true):
        tv = to_numpy(true)
        tc = to_numpy(true[:, 0])
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
        mbd_sample = mean_bucket_distance(ic, tc)
        mbd_bal = balanced_mean_bucket_distance(ic, tc, num_classes=ycat)
        alpha = 0.3
        mbd = (1.0 - alpha) * mbd_sample + alpha * mbd_bal
        accr = balanced_within_one_accuracy(ic, tc, num_classes=ycat)
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
