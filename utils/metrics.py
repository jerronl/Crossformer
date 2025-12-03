import numpy as np
import torch

metric_names = [
    [
        "mae",
        "mse",
        "rmse",
        "mape",
        "mspe",
    ],
    [
        "mae",
        "mse",
        "rmse",
        "rps",
        "smooth",
        "mode_err",
        "emd",
        "tail_err",
        "class_sep",
    ],
]


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


def RPS(probs, true_labels):
    n_classes = probs.shape[-1]
    pred_cdf = np.cumsum(probs, axis=-1)
    true_labels = true_labels.astype(int)
    true_onehot = np.eye(n_classes)[true_labels]
    true_cdf = np.cumsum(true_onehot, axis=-1)
    rps = np.mean(np.sum((pred_cdf - true_cdf) ** 2, axis=-1))
    return rps


def class_separation(probs, true_labels, num_classes=None):
    true_labels = true_labels.astype(int)
    if num_classes is None:
        num_classes = int(true_labels.max()) + 1
    means = []
    for c in range(num_classes):
        mask = true_labels == c
        if not np.any(mask):
            continue
        means.append(probs[mask].mean(axis=0))
    if len(means) < 2:
        return 0.0
    means = np.stack(means, axis=0)
    diffs = []
    n = means.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            diff = means[i] - means[j]
            cdf_diff = np.cumsum(diff)
            d = np.mean(np.abs(cdf_diff))
            diffs.append(d)
    if not diffs:
        return 0.0
    return float(np.mean(diffs))


def make_metric(ycat):
    def metric_cat(pred, true):
        tv = to_numpy(true)
        tc = to_numpy(true[:, 0]).astype(int) - 1
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
        probs = softmax(ic, axis=-1)
        rps = RPS(probs, tc)
        diff = probs[:, 1:] - probs[:, :-1]
        smooth = np.mean(diff * diff)
        mode = probs.argmax(axis=-1)
        mode_err = np.mean(np.abs(mode - tc))
        n_classes = probs.shape[-1]
        pred_cdf = np.cumsum(probs, axis=-1)
        true_onehot = np.eye(n_classes)[tc]
        true_cdf = np.cumsum(true_onehot, axis=-1)
        emd = np.mean(np.abs(pred_cdf - true_cdf))
        cols = np.arange(n_classes)[None, :]
        tpos = tc[:, None]
        left3 = np.clip(tpos - 3, 0, n_classes - 1)
        right3 = np.clip(tpos + 3, 0, n_classes - 1)
        center_mask = (cols >= left3) & (cols <= right3)
        prob_center = (probs * center_mask).sum(axis=1)
        tail_global = 1.0 - prob_center
        left_mask = cols < (tpos - 1)
        right_mask = cols > (tpos + 1)
        left_mass = (probs * left_mask).sum(axis=1)
        right_mass = (probs * right_mask).sum(axis=1)
        tail_asym = np.abs(left_mass - right_mass)
        tail_err = float(np.mean(tail_global + tail_asym))
        class_sep = class_separation(probs, tc, num_classes=ycat)
        return (
            mae,
            mse,
            rmse,
            rps,
            smooth,
            mode_err,
            emd,
            tail_err,
            class_sep,
        )

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
        return mae, mse, rmse, mape, mspe

    return (metric_cat, metric_names[1]) if ycat > 0 else (metric, metric_names[0])
