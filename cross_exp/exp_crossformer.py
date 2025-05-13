import os
import time
import pickle

import warnings
import numpy as np

import torch
from torch import nn, optim, isnan
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from data.data_loader import DatasetMTS
from cross_exp.exp_basic import Exp_Basic
from cross_models.cross_former import Crossformer
from utils.tools import EarlyStopping, print_color
from utils.metrics import make_metric

warnings.filterwarnings("ignore")


def nanstd(x: torch.Tensor, dim=0, unbiased=False):
    mask = ~torch.isnan(x)
    count = mask.sum(dim)
    mean = torch.nanmean(x, dim)
    diff = (x - mean).pow(2)
    diff[~mask] = 0
    var = diff.sum(dim) / (count - (1 if unbiased else 0)).clamp(min=1)
    return torch.sqrt(var)


def fuzzy_accuracy(pred_logits, true_labels):
    pred_class = pred_logits.argmax(dim=-1)
    diff = (pred_class - true_labels).abs()
    score = torch.where(diff == 0, 1.0, torch.where(diff == 1, 0.25, 0.0))
    return score.mean().item()


def cross_entropy_with_nans(ic, tc):
    # ic: (B, 1, C), tc: (B,)
    mc = ~torch.isnan(ic).any(dim=2).squeeze(1)  # mask of valid samples
    if mc.sum() == 0:
        return mc.sum() * 0 + 1e-8
    else:
        # valid indices
        tc_valid = tc[mc]  # (valid,)
        log_probs_valid = F.log_softmax(ic[mc][:, 0, :], dim=-1)  # (valid, C)

        # exact match
        exact_loss = -log_probs_valid[
            torch.arange(log_probs_valid.size(0)), tc_valid
        ].mean()

        # adjacent categories
        num_classes = ic.size(-1)
        adjacents = [torch.clamp(tc_valid + i, 0, num_classes - 1) for i in [1, -1]]

        adjacent_losses = [
            -log_probs_valid[torch.arange(log_probs_valid.size(0)), adj].mean()
            for adj in adjacents
        ]
        total_loss = exact_loss + sum(adjacent_losses) / (4.0 * len(adjacent_losses))

    # Add penalty for missing samples
    return total_loss + (~mc).sum() / tc.numel()


class Exp_crossformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_crossformer, self).__init__(args)
        self.ycat = self.model = None
        self.checkpoint = {}
        self.loss_logits = nn.Parameter(
            torch.log(
                torch.tensor(
                    [args.lambda_mse, args.lambda_huber, 0.1, 0.1],
                    dtype=torch.float32,
                )
            )
        )
        self.log_sigma_mu = nn.Parameter(torch.zeros(()))
        self.log_sigma_q90 = nn.Parameter(torch.zeros(()))
        self.log_delta = nn.Parameter(torch.log(torch.tensor(args.delta)))
        self.weight = len(self.loss_logits) * 0.1 + args.weight + 1.2
        self.weight = 1 / self.weight, 0.1 / self.weight, args.weight / self.weight
        self.alpha = 1 / 6
        self.step = args.step

    def build_model(self, data):
        model = Crossformer(
            data.data_dim,
            data.out_dim,
            data.ycat,
            self.args.in_len,
            self.args.out_len,
            self.args.seg_len,
            data.sect,
            data.sp,
            self.args.win_size,
            self.args.factor,
            self.args.d_model,
            self.args.d_ff,
            self.args.n_heads,
            self.args.e_layers,
            self.args.dropout,
            self.args.baseline,
            self.device,
            [torch.tensor(d, dtype=torch.float32) for d in data[:2][0]],
        ).float()
        self.ycat = data.ycat
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        self.model = model.to(self.device)

    def _get_data(self, flag, data=None, data_path=None, scaler=None, data_split=None):
        args = self.args

        drop_last = False
        batch_size = args.batch_size
        shuffle_flag = flag == "train"
        data_set = DatasetMTS(
            root_path=args.root_path,
            data_path=data_path if data_path is not None else args.data_path,
            data_name=data or args.data,
            flag=flag,
            in_len=args.in_len,
            data_split=data_split or args.data_split,
            query=args.query,
            scaler=scaler,
        )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, ycat):
        tau = 0.7

        def cross_entropy_mse_loss_with_nans(input, target):
            assert input[0].shape[1] == 1
            _, tc = target
            _, _, ic = input
            return cross_entropy_with_nans(ic, tc) + cross_mse_loss_with_nans(
                input, target
            )

        def cross_mse_loss_with_nans(input, target):
            pred_mu, _, _ = input
            assert pred_mu.shape[1] == 1
            tv, _ = target
            mi = isnan(pred_mu)
            mask = isnan(tv) | mi
            valid_mu = pred_mu[~mask]
            if valid_mu.numel() == 0:
                return valid_mu.sum() * 0 + 1e-8
            valid_tv = tv[~mask]
            delta = torch.exp(self.log_delta)
            weights = F.softmax(self.loss_logits, dim=0) * self.weight[1] * 2
            # entropy_reg = (weights * torch.log(weights + 1e-8)).sum()
            # weights = weights + torch.std(weights) / len(weights)
            # weights = weights / weights.sum() * 0.1

            # loss_huber = F.smooth_l1_loss(valid_mu, valid_tv, beta=delta)
            u = valid_mu - valid_tv
            abs_u = torch.abs(u)
            # L_δ(u) = { 0.5 u²/δ   if |u|<δ;   |u| - 0.5δ  otherwise }
            loss_huber = torch.where(
                abs_u < delta, 0.5 * u**2 / delta, abs_u - 0.5 * delta
            ).mean()
            loss_mse = (
                (1 + (self.alpha * valid_tv.abs()).clamp(0, 0.5)) * u.pow(2)
            ).mean()
            mask_pos = (tv < 0.1) | mask
            tv_pos = tv.clone()
            mu_pos = pred_mu.clone()
            tv_pos[mask_pos] = float("nan")
            mu_pos[mask_pos] = float("nan")
            counts = (~mask_pos).sum(dim=0)  # (C,)

            small_cols = counts < 5
            tv_pos[:, small_cols] = float("nan")
            mu_pos[:, small_cols] = float("nan")

            std_tv_pos = nanstd(tv_pos, dim=0)  # (C,)
            std_mu_pos = nanstd(mu_pos, dim=0)  # (C,)
            u = torch.nan_to_num(std_tv_pos - std_mu_pos)
            loss_q90_pos = torch.mean(torch.max(tau * u, (tau - 1) * u))
            sigma_mu = torch.exp(self.log_sigma_mu).clamp(min=1e-3, max=1e3)
            sigma_q90 = torch.exp(self.log_sigma_q90).clamp(min=1e-3, max=1e3)

            loss_mu_part = loss_mse / (2 * sigma_mu**2) + torch.log(sigma_mu)
            loss_q90_pos = loss_q90_pos / (2 * sigma_q90**2) + torch.log(sigma_q90)
            # Compute variance of predictions and targets
            variance_tv = nanstd(tv)
            variance_iv = nanstd(pred_mu)

            # Variance loss (maximize variance matching)
            variance_loss = (
                (variance_iv - variance_tv).pow(2)
                + (variance_tv / (1e-8 + variance_iv) - 1).pow(2) * 0.01
            ).mean()
            return (
                torch.sqrt(
                    torch.clamp(
                        (weights[0] + self.weight[0]) * loss_mu_part
                        + (weights[1] + self.weight[1]) * loss_huber
                        + (weights[2] + self.weight[2]) * variance_loss,
                        min=1e-8,
                    )
                )
                + (weights[3] + self.weight[1]) * loss_q90_pos
                + mi.sum() / target[0].numel()
            )

        return (
            cross_entropy_mse_loss_with_nans if ycat > 0 else cross_mse_loss_with_nans
        )

    def vali(self, vali_data, vali_loader):
        self.model.eval()
        pred, y, ic, yc = [], [], [], []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                (pv, _, pc), (tv, tc) = self._process_one_batch(
                    vali_data, batch_x, batch_y, inverse=True
                )
                pred.append(pv)
                ic.append(pc)
                y.append(tv)
                yc.append(tc)
        pred = np.concatenate(pred)
        y = np.concatenate(y)
        mse = np.mean((1 + np.minimum(self.alpha * np.abs(y), 0.5)) * (pred - y) ** 2)

        var_y = np.std(y, axis=0)
        var_p = np.std(pred, axis=0)
        var_abs = np.mean((var_p - var_y) ** 2)
        # var_rel = np.mean((var_y / (var_p + 1e-8) - 1) ** 2)
        if ic[0] is None:
            ce = 0
        else:
            ic, yc = [torch.from_numpy(np.concatenate(x)) for x in (ic, yc)]
            ce = (
                cross_entropy_with_nans(ic, yc).item()
                + 10
                - fuzzy_accuracy(ic, tc) * 10
            )

        # self.model.train()
        return mse + var_abs + ce

    def train(self, setting, data):
        checkpoint = data_split = None
        key = setting + data
        path = os.path.join(self.args.checkpoints, key)
        if not os.path.exists(path):
            os.makedirs(path)
        if self.args.resume:
            best_model_path = path + "/" + "checkpoint.pth"
            try:
                checkpoint = torch.load(best_model_path, weights_only=False)
                if len(checkpoint) > 1:
                    data_split = checkpoint[0][4]
            except (
                FileNotFoundError,
                RuntimeError,
                IndexError,
                pickle.UnpicklingError,
            ) as e:
                print_color(91, "failed to load", e, best_model_path)
        train_data, train_loader = self._get_data(
            flag="train", data=data, data_split=data_split
        )
        vali_data, vali_loader = self._get_data(
            flag="val", data=data, data_split=data_split
        )
        test_data, test_loader = self._get_data(
            flag="test", data=data, data_split=data_split
        )
        self.build_model(train_data)

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.ycat)
        score = None
        spoch = 0
        if checkpoint is not None:
            try:
                self.load_state_dict(checkpoint[0][0])
                model_optim.load_state_dict(checkpoint[0][1])
                score = self.vali(vali_data, vali_loader)
                # score = abs(checkpoint[1])
                spoch = checkpoint[0][2]
                print_color(
                    93,
                    f"suc to load. score {score} epoch {spoch} from:",
                    best_model_path,
                )
            except (
                RuntimeError,
                IndexError,
                pickle.UnpicklingError,
            ) as e:
                print_color(91, "failed to load", e, best_model_path)
        early_stopping = EarlyStopping(
            lradj=self.args.lradj,
            learning_rate=self.args.learning_rate,
            patience=self.args.patience,
            verbose=True,
            best_score=score,
            step=self.step,
        )
        for epoch in range(spoch, self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(train_data, batch_x, batch_y)
                loss = criterion(pred, true)
                assert ~isnan(loss)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0:.4g}, epoch: {1} | loss: {2:.4g}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4g}s/iter; left time: {:.4g}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print(
                "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time),
            )
            train_loss = np.average(train_loss)
            vali_loss = self.vali(
                vali_data,
                vali_loader,
            )
            test_loss = self.vali(
                test_data,
                test_loader,
            )

            print(
                "Epoch: {0:.3e}, Steps: {1} | Train Loss: {2:.4g} Vali Loss: {3:.4g} Test Loss: {4:.4g}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(
                vali_loss,
                (
                    self.state_dict(),
                    model_optim.state_dict(),
                    epoch + 1,
                    train_data.scaler,
                    train_data.data_split,
                ),
                path,
            )
            if early_stopping.early_stop:
                print_color(95, "Early stopping")
                break

            if early_stopping.counter > 1 and early_stopping.adjust_learning_rate(
                model_optim
            ):
                best_model_path = path + "/" + "checkpoint.pth"
                checkpoint = list(torch.load(best_model_path, weights_only=False))
                self.load_state_dict(checkpoint[0][0])
                model_optim.load_state_dict(checkpoint[0][1])

        best_model_path = path + "/" + "checkpoint.pth"
        checkpoint = list(torch.load(best_model_path, weights_only=False))
        self.load_state_dict(checkpoint[0][0])
        checkpoint[0] = list(checkpoint[0])
        checkpoint[0][0] = self.state_dict()
        torch.save(checkpoint, path + "/checkpoint.pth")
        self.checkpoint[key] = (self.model, checkpoint[0][3], checkpoint[0][4])
        torch.save(
            self.checkpoint[key],
            path + "/crossformer.pkl",
        )

        return self.model

    def test(
        self,
        setting="model",
        data="vols",
        save_pred=False,
        inverse=False,
        data_path=None,
        run_metric=True,
    ):
        key = setting + data
        if key not in self.checkpoint:
            best_model_path = (
                os.path.join(self.args.checkpoints, key) + "/crossformer.pkl"
            )
            try:
                self.checkpoint[key] = torch.load(best_model_path, weights_only=False)
                print_color(93, "suc to load", best_model_path)
            except (
                FileNotFoundError,
                RuntimeError,
                IndexError,
                pickle.UnpicklingError,
            ) as e:
                print_color(91, "failed to load", e, best_model_path)
        self.model = self.checkpoint[key][0]
        test_data, test_loader = self._get_data(
            data=data,
            flag="test",
            scaler=self.checkpoint[key][1],
            data_path=data_path,
            data_split=(
                self.checkpoint[key][2] if len(self.checkpoint[key]) > 2 else None
            ),
        )

        self.model.eval()

        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        metric = make_metric(test_data.ycat)

        pred, y, ic, yc = [], [], [], []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                (pv, _, pc), (tv, tc) = self._process_one_batch(
                    test_data, batch_x, batch_y, inverse=inverse
                )
                pred.append(pv)
                ic.append(pc)
                y.append(tv)
                yc.append(tc)
                batch_size = tv.shape[0]
                instance_num += batch_size

        pred, y = [np.concatenate(x) for x in (pred, y)]
        ic, yc = (
            [None, None] if ic[0] is None else [np.concatenate(x) for x in (ic, yc)]
        )

        if run_metric:
            metrics_mean = np.array(metric(pred, y, ic, yc))
            # result save
            folder_path = "./results/" + setting + "/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe, accr = metrics_mean
            print_color(
                94,
                f"mae:{mae:.3e}, mse:{mse:.3e}, rmse:{rmse:.3e}, mape:{mape:.3e}, mspe:{mspe:.3e}, accr:{accr}",
            )

            np.save(
                folder_path + f"mmae{mae:.3e}, accr{accr}.npy",
                np.array([mae, mse, rmse, mape, mspe, accr]),
            )
            # if save_pred:
            #     preds = np.concatenate(preds, axis=0)
            #     trues = np.concatenate(trues, axis=0)
            #     np.save(folder_path + "pred.npy", preds)
            #     np.save(folder_path + "true.npy", trues)
        else:
            metrics_mean = ()

        return (
            np.squeeze(
                (pred if ic is None else np.concatenate([pred, ic], axis=-1)), axis=1
            ),
            np.squeeze(y, axis=1),
            metrics_mean,
        )

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse=False):
        batch_x = [x.float().to(self.device) for x in batch_x]
        batch_y = batch_y[0].float().to(self.device), batch_y[1].long().to(self.device)

        outputs = self.model(batch_x)

        if inverse:
            return self._inverse(dataset_object, outputs, batch_y)
        return outputs, batch_y

    def _inverse(self, dataset_object, outputs, batch_y):
        outputs = dataset_object.inverse_transform(outputs)
        batch_y = dataset_object.inverse_transform(batch_y)
        return outputs, batch_y
