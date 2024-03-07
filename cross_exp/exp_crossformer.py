from data.data_loader import DatasetMTS
from cross_exp.exp_basic import Exp_Basic
from cross_models.cross_former import Crossformer

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import make_metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import os
import time
import json
import pickle

import warnings

warnings.filterwarnings("ignore")


class Exp_crossformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_crossformer, self).__init__(args)
        self.ycat = self.model = None

    def build_model(self, data):
        model = Crossformer(
            data.data_dim,
            data.out_dim,
            data.ycat,
            self.args.in_len,
            self.args.out_len,
            self.args.seg_len,
            self.args.win_size,
            self.args.factor,
            self.args.d_model,
            self.args.d_ff,
            self.args.n_heads,
            self.args.e_layers,
            self.args.dropout,
            self.args.baseline,
            self.device,
        ).float()
        self.ycat = data.ycat
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        self.model = model.to(self.device)

    def _get_data(self, flag, data=None, data_path=None, scaler=None):
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
            data_path=data_path or args.data_path,
            data_name=data or args.data,
            flag=flag,
            in_len=args.in_len,
            data_split=args.data_split,
            cutday=args.cutday,
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
        mse, cel = nn.MSELoss(), nn.CrossEntropyLoss()

        def cross_entropy_mse_loss_with_nans(input, target):
            assert input.shape[1] == 1
            tv, tc = target
            iv, ic = input[:, :, :-ycat], input[:, 0, -ycat:]
            mi = torch.isnan(iv)
            mask = torch.isnan(tv) | mi
            mc = torch.isnan(ic).any(dim=1)
            return (
                (mse(iv[~mask], tv[~mask]) ** 0.5) * 10
                + cel(ic[~mc], tc[~mc])
                + (mi.sum() + mc.sum()) / target[0].numel()
            )

        def cross_mse_loss_with_nans(input, target):
            assert input.shape[1] == 1
            tv, _ = target
            iv = input
            mi = torch.isnan(iv)
            mask = torch.isnan(tv) | mi
            return (mse(iv[~mask], tv[~mask]) ** 0.5) * 10 + mi.sum() / target[
                0
            ].numel()

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
        train_data, train_loader = self._get_data(flag="train", data=data)
        vali_data, vali_loader = self._get_data(flag="val", data=data)
        test_data, test_loader = self._get_data(flag="test", data=data)
        self.build_model(train_data)

        path = os.path.join(self.args.checkpoints, setting + data)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.ycat)
        score = None
        spoch = 0
        if self.args.resume:
            best_model_path = path + "/" + "checkpoint.pth"
            try:
                checkpoint = torch.load(best_model_path)
                self.model.load_state_dict(checkpoint[0][0])
                model_optim.load_state_dict(checkpoint[0][1])
                score = abs(checkpoint[1])
                spoch = checkpoint[0][2]
                print(
                    f"\033[92msuc to load. score {score} epoch {spoch} from:",
                    best_model_path,
                    "\033[0m",
                )
            except (
                FileNotFoundError,
                RuntimeError,
                IndexError,
                pickle.UnpicklingError,
            ) as e:
                print("\033[91mfailed to load", e, best_model_path, "\033[0m")
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, best_score=score
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
                assert ~torch.isnan(loss)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
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
                ),
                path,
            )
            if early_stopping.early_stop:
                print("\033[95mEarly stopping\033[0m")
                break

            if early_stopping.counter > 1:
                adjust_learning_rate(model_optim, epoch + 1 - spoch, self.args)
            else:
                spoch += 1

        best_model_path = path + "/" + "checkpoint.pth"
        checkpoint = list(torch.load(best_model_path))
        self.model.load_state_dict(checkpoint[0][0])
        checkpoint[0] = list(checkpoint[0])
        checkpoint[0][0] = (
            self.model.module.state_dict()
            if isinstance(self.model, DataParallel)
            else self.model.state_dict()
        )
        torch.save(checkpoint, path + "/" + "checkpoint.pth")
        torch.save((self.model, checkpoint[0][3]), path + "/" + "crossformer.pkl")

        return self.model

    def test(self, setting, data, save_pred=False, inverse=False, data_path=None):
        best_model_path = (
            os.path.join(self.args.checkpoints, setting + data) + "/crossformer.pkl"
        )
        try:
            checkpoint = torch.load(best_model_path)
            self.model = checkpoint[0]
            print("\033[92msuc to load", best_model_path, "\033[0m")
        except (
            FileNotFoundError,
            RuntimeError,
            IndexError,
            pickle.UnpicklingError,
        ) as e:
            print("\033[91mfailed to load", e, best_model_path, "\033[0m")
        test_data, test_loader = self._get_data(
            data=data, flag="test", scaler=checkpoint[1], data_path=data_path
        )

        self.model.eval()

        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        metric = make_metric(test_data.ycat)

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, inverse
                )
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred, true)) * batch_size
                metrics_all.append(batch_metric)
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

        metrics_all = np.stack(metrics_all, axis=0)
        metrics_mean = metrics_all.sum(axis=0) / instance_num

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, accr = metrics_mean
        print("\033[93mmse:{}, mae:{}".format(mse, mae), metrics_mean, "\033[0m")

        np.save(
            folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe, accr])
        )
        if save_pred:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            np.save(folder_path + "pred.npy", preds)
            np.save(folder_path + "true.npy", trues)

        return preds, trues

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse=False):
        batch_x = [x.float().to(self.device) for x in batch_x]
        batch_y = batch_y[0].float().to(self.device), batch_y[1].type(
            torch.LongTensor
        ).to(self.device)

        outputs = self.model(batch_x)

        if inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)

        return outputs, batch_y

    def eval(self, setting, save_pred=False, inverse=False):
        # evaluate a saved model
        args = self.args
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag="test",
            size=[args.in_len, args.out_len],
            data_split=args.data_split,
            scale=True,
            scale_statistic=args.scale_statistic,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

        self.model.eval()

        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        metric = make_metric(data_set.ycat)

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(data_loader):
                pred, true = self._process_one_batch(
                    data_set, batch_x, batch_y, inverse
                )
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = (
                    np.array(
                        metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
                    )
                    * batch_size
                )
                metrics_all.append(batch_metric)
                if save_pred:
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis=0)
        metrics_mean = metrics_all.sum(axis=0) / instance_num

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print("mse:{}, mae:{}".format(mse, mae))

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        if save_pred:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            np.save(folder_path + "pred.npy", preds)
            np.save(folder_path + "true.npy", trues)

        return mae, mse, rmse, mape, mspe
