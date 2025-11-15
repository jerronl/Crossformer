import numpy as np

# from sklearn.metrics import accuracy_score


def fuzzy_accuracy(pred_logits, true_labels, alpha=np.log(4.0)):
    # 统一转成 numpy，防止 tensor/numpy 混着报错
    pred_logits_np = (
        pred_logits.detach().cpu().numpy()
        if hasattr(pred_logits, "detach")
        else np.asarray(pred_logits)
    )
    true_labels_np = (
        true_labels.detach().cpu().numpy()
        if hasattr(true_labels, "detach")
        else np.asarray(true_labels)
    )

    # 取分类结果
    pred_class = np.argmax(pred_logits_np, axis=-1)   # (...,)
    diff = np.abs(pred_class - true_labels_np)        # (...,)

    # 距离加权分数：d=0 -> 1, d=1 -> 0.25, d=2 -> ~0.06, d越大越接近0
    score = np.exp(-alpha * diff)

    return score.mean()

def mean_bucket_distance(pred_logits, true_labels):
    pred_logits_np = (
        pred_logits.detach().cpu().numpy()
        if hasattr(pred_logits, "detach")
        else np.asarray(pred_logits)
    )
    true_labels_np = (
        true_labels.detach().cpu().numpy()
        if hasattr(true_labels, "detach")
        else np.asarray(true_labels)
    )

    pred_class = np.argmax(pred_logits_np, axis=-1)
    diff = np.abs(pred_class - true_labels_np)
    return diff.mean()


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
    return np.mean(np.square((pred - true) / true))


def make_metric(ycat):
    def metric_cat(pred, true):
        tv, tc = (
            true
            if isinstance(true[0], np.ndarray)
            else (true[0].detach().cpu().numpy(), true[1].detach().cpu().numpy())
        )
        iv, ic = (
            (pred[:, :-ycat], pred[:, -ycat:])
            if isinstance(pred, np.ndarray)
            else (
                pred[:, :, :-ycat].detach().cpu().numpy(),
                pred[:, 0, -ycat:].detach().cpu().numpy(),
            )
        )
        mae = MAE(iv, tv)
        mse = MSE(iv, tv)
        rmse = RMSE(iv, tv)
        mape = MAPE(iv, tv)
        mspe = mean_bucket_distance(ic, tc)
        accr = fuzzy_accuracy(ic, tc)

        return mae, mse, rmse, mape, mspe, accr

    def metric(pred, true):

        tv, _ = (
            true
            if isinstance(true[0], np.ndarray)
            else (true[0].detach().cpu().numpy(), true[1].detach().cpu().numpy())
        )
        iv = pred if isinstance(pred, np.ndarray) else pred.detach().cpu().numpy()
        mae = MAE(iv, tv)
        mse = MSE(iv, tv)
        rmse = RMSE(iv, tv)
        mape = MAPE(iv, tv)
        mspe = MSPE(iv, tv)

        return mae, mse, rmse, mape, mspe, -1

    return metric_cat if ycat > 0 else metric
