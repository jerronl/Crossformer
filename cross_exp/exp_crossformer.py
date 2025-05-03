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


class Exp_crossformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_crossformer, self).__init__(args)
        self.ycat = self.model = None
        self.checkpoint = {}
        self.loss_logits = nn.Parameter(
            torch.log(
                torch.tensor(
                    [args.weight, args.lambda_mse, args.lambda_huber, 0.1],
                    dtype=torch.float32,
                )
            )
        )
        self.log_sigma_mu = nn.Parameter(torch.zeros(()))
        self.log_sigma_q90 = nn.Parameter(torch.zeros(()))
        self.log_delta = nn.Parameter(torch.log(torch.tensor(args.delta)))

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

        if flag == "test":
            shuffle_flag = False
            drop_last = False
            batch_size = args.batch_size
        else:
            shuffle_flag = True
            drop_last = False
            batch_size = args.batch_size
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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, ycat):
        tau = 0.9

        def cross_entropy_mse_loss_with_nans(input, target):
            assert input[0].shape[1] == 1
            _, tc = target
            iv, ic = (input[0][:, :, :-ycat], input[1][:, :, :-ycat]), input[0][
                :, 0, -ycat:
            ]
            mc = isnan(ic).any(dim=1)
            adjacents = [
                torch.clamp(tc[~mc] + i, 0, ic.size(1) - 1) for i in [1, -1]
            ] + [
                torch.clamp(ic.size(1) - tc[~mc] + i, 0, ic.size(1) - 1)
                for i in [
                    -1,
                ]  # -1, -2]
            ]
            log_probs_valid = F.log_softmax(
                ic[~mc], dim=-1
            )  # Shape: (batch_size, num_classes)
            tc_valid = tc[~mc]  # Shape: (valid_batch_size,)

            # Exact match loss
            exact_match_loss = -log_probs_valid[
                torch.arange(log_probs_valid.size(0)), tc_valid
            ].mean()

            # Compute adjacent losses
            adjacent_losses = [
                -log_probs_valid[torch.arange(log_probs_valid.size(0)), adjacent].mean()
                for adjacent in adjacents
            ]
            # Combine losses
            total_loss = exact_match_loss + sum(adjacent_losses) / 4.0 / len(
                adjacent_losses
            )
            return (
                total_loss
                + cross_mse_loss_with_nans(iv, target)
                + mc.sum() / target[0].numel()
            )

        def cross_mse_loss_with_nans(input, target):
            assert input[0].shape[1] == 1
            tv, _ = target
            # iv = input
            # mi = isnan(iv)
            # mask = isnan(tv) | mi
            # valid_iv = iv[~mask]
            # valid_tv = tv[~mask]

            # # Compute MSE
            # mse_loss = (valid_iv - valid_tv).pow(2).mean()

            # # Compute variance of predictions and targets
            # variance_tv = ((valid_tv - valid_tv.mean()).pow(2)).mean()
            # variance_iv = ((valid_iv - valid_iv.mean()).pow(2)).mean()

            # # Variance loss (maximize variance matching)
            # variance_loss = (variance_iv - variance_tv).pow(2) + (
            #     variance_tv / (1e-8 + variance_iv) - 1
            # ).pow(2) * 0.01

            # return (mse_loss + weight * variance_loss) ** 0.5 * 10 + mi.sum() / target[
            #     0
            # ].numel()

            pred_mu, pred_q90 = input
            mi = isnan(pred_mu) | isnan(pred_q90)
            mask = isnan(tv) | mi
            valid_mu = pred_mu[~mask]
            valid_q90 = pred_q90[~mask]
            valid_tv = tv[~mask]
            delta = torch.exp(self.log_delta)  # 保证 >0
            weights = F.softmax(self.loss_logits, dim=0)
            entropy_reg = (weights * torch.log(weights + 1e-8)).sum()
            weights += weights.max() / len(weights)
            weights /= weights.sum()

            # loss_huber = F.smooth_l1_loss(valid_mu, valid_tv, beta=delta)
            u = valid_mu - valid_tv
            abs_u = torch.abs(u)
            # L_δ(u) = { 0.5 u²/δ   if |u|<δ;   |u| - 0.5δ  otherwise }
            loss_huber = torch.where(
                abs_u < delta, 0.5 * u**2 / delta, abs_u - 0.5 * delta
            ).mean()
            loss_mse = F.mse_loss(valid_mu, valid_tv)
            u = valid_tv - valid_q90
            loss_q90 = torch.mean(torch.max(tau * u, (tau - 1) * u))
            mask_pos = valid_tv > 0
            u = valid_tv[mask_pos] - valid_q90[mask_pos]
            loss_q90 += torch.mean(torch.max(tau * u, (tau - 1) * u))
            sigma_mu = torch.exp(self.log_sigma_mu)
            sigma_q90 = torch.exp(self.log_sigma_q90)

            loss_mu_part = loss_mse / (2 * sigma_mu**2) + torch.log(sigma_mu)
            loss_q90_part = loss_q90 / (2 * sigma_q90**2) + torch.log(sigma_q90)
            # Compute variance of predictions and targets
            variance_tv = ((valid_tv - valid_tv.mean()).pow(2)).mean()
            variance_iv = ((valid_mu - valid_mu.mean()).pow(2)).mean()

            # Variance loss (maximize variance matching)
            variance_loss = (variance_iv - variance_tv).pow(2) + (
                variance_tv / (1e-8 + variance_iv) - 1
            ).pow(2) * 0.01
            return (
                weights[2] * loss_huber
                + weights[1] * loss_mu_part
                + weights[0] * variance_loss
                + weights[1] * loss_q90_part
            ) ** 0.5 + weights[3] * entropy_reg

        return (
            cross_entropy_mse_loss_with_nans if ycat > 0 else cross_mse_loss_with_nans
        )

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                pred, true = self._process_one_batch(vali_data, batch_x, batch_y)
                loss = criterion(pred, true)
                total_loss.append(loss.detach().item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

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
                self.model.load_state_dict(checkpoint[0][0])
                model_optim.load_state_dict(checkpoint[0][1])
                score = abs(checkpoint[1])
                spoch = checkpoint[0][2]
                print_color(
                    94,
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
                        "\titers: {0:.3f}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print(
                "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time),
                "weight",
                self.loss_logits,
            )
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0:.3f}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(
                vali_loss,
                (
                    self.model.state_dict(),
                    model_optim.state_dict(),
                    epoch,
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
                self.model.load_state_dict(checkpoint[0][0])
                model_optim.load_state_dict(checkpoint[0][1])

        best_model_path = path + "/" + "checkpoint.pth"
        checkpoint = list(torch.load(best_model_path, weights_only=False))
        self.model.load_state_dict(checkpoint[0][0])
        checkpoint[0] = list(checkpoint[0])
        checkpoint[0][0] = (
            self.model.module.state_dict()
            if isinstance(self.model, DataParallel)
            else self.model.state_dict()
        )
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
                print_color(94, "suc to load", best_model_path)
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

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                (pred, _), true = self._process_one_batch(test_data, batch_x, batch_y)
                batch_size = pred.shape[0]
                instance_num += batch_size
                if run_metric:
                    batch_metric = np.array(metric(pred, true)) * batch_size
                    metrics_all.append(batch_metric)
                if inverse:
                    pred, true = self._inverse(test_data, pred, true)
                if save_pred:
                    preds.append(
                        pred
                        if isinstance(pred, np.ndarray)
                        else pred.detach().cpu().numpy()
                    )
                    trues.append(
                        true[0]
                        if isinstance(true[0], np.ndarray)
                        else true[0].detach().cpu().numpy()
                    )

        if run_metric:
            metrics_all = np.stack(metrics_all, axis=0)
            metrics_mean = metrics_all.sum(axis=0) / instance_num

            # result save
            folder_path = "./results/" + setting + "/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe, accr = metrics_mean
            print_color(
                93,
                f"mae:{mae:.3f}, mse:{mse:.3f}, rmse:{rmse:.3f}, mape:{mape:.3f}, mspe:{mspe:.3f}, accr:{accr}",
            )

            np.save(
                folder_path + f"mmae{mae:.3f}, accr{accr}.npy",
                np.array([mae, mse, rmse, mape, mspe, accr]),
            )
            # if save_pred:
            #     preds = np.concatenate(preds, axis=0)
            #     trues = np.concatenate(trues, axis=0)
            #     np.save(folder_path + "pred.npy", preds)
            #     np.save(folder_path + "true.npy", trues)
        else:
            metrics_mean = ()

        return np.concatenate(preds), np.concatenate(trues), metrics_mean

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse=False):
        batch_x = [x.float().to(self.device) for x in batch_x]
        batch_y = batch_y[0].float().to(self.device), batch_y[1].type(
            torch.LongTensor
        ).to(self.device)

        outputs = self.model(batch_x)

        if inverse:
            return self._inverse(dataset_object, outputs, batch_y)
        return outputs, batch_y

    def _inverse(self, dataset_object, outputs, batch_y):
        if dataset_object.ycat > 0:
            outputs[:, :, -dataset_object.ycat :] = F.softmax(
                outputs[:, :, -dataset_object.ycat :], 2
            )
        outputs = dataset_object.inverse_transform(outputs)
        batch_y = dataset_object.inverse_transform(batch_y)
        return outputs, batch_y

    # def eval(self, setting, save_pred=False, inverse=False):
    #     # evaluate a saved model
    #     args = self.args
    #     data_set = Dataset_MTS(
    #         root_path=args.root_path,
    #         data_path=args.data_path,
    #         flag="test",
    #         size=[args.in_len, args.out_len],
    #         data_split=args.data_split,
    #         scale=True,
    #         scale_statistic=args.scale_statistic,
    #     )

    #     data_loader = DataLoader(
    #         data_set,
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         num_workers=args.num_workers,
    #         drop_last=False,
    #     )

    #     self.model.eval()

    #     preds = []
    #     trues = []
    #     metrics_all = []
    #     instance_num = 0
    #     metric = make_metric(data_set.ycat)

    #     with torch.no_grad():
    #         for i, (batch_x, batch_y) in enumerate(data_loader):
    #             pred, true = self._process_one_batch(
    #                 data_set, batch_x, batch_y, inverse
    #             )
    #             batch_size = pred.shape[0]
    #             instance_num += batch_size
    #             batch_metric = (
    #                 np.array(
    #                     metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
    #                 )
    #                 * batch_size
    #             )
    #             metrics_all.append(batch_metric)
    #             if save_pred:
    #                 preds.append(pred.detach().cpu().numpy())
    #                 trues.append(true.detach().cpu().numpy())

    #     metrics_all = np.stack(metrics_all, axis=0)
    #     metrics_mean = metrics_all.sum(axis=0) / instance_num

    #     # result save
    #     folder_path = "./results/" + setting + "/"
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     mae, mse, rmse, mape, mspe = metrics_mean
    #     print("mse:{:.3f}, mae:{}".format(mse, mae))

    #     np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
    #     if save_pred:
    #         preds = np.concatenate(preds, axis=0)
    #         trues = np.concatenate(trues, axis=0)
    #         np.save(folder_path + "pred.npy", preds)
    #         np.save(folder_path + "true.npy", trues)

    #     return mae, mse, rmse, mape, mspe
