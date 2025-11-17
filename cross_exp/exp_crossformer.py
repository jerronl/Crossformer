import os
import sys
import traceback
import time
import pickle

import warnings
import numpy as np

import torch.profiler
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.data_loader import DatasetMTS
from cross_exp.exp_basic import Exp_Basic
from cross_models.cross_former import Crossformer
from utils.tools import EarlyStopping, print_color, format_nested
from utils.metrics import make_metric


warnings.filterwarnings("ignore")


def match_lambda(
    v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8, base: float = 2.0
) -> torch.Tensor:
    v1 = torch.as_tensor(v1)
    v2 = torch.as_tensor(v2)
    ratio = v1.detach() / (v2.detach() + eps)
    log_ratio = torch.log(ratio)
    quant = torch.floor(log_ratio / base) * base
    return torch.exp(quant)


class Exp_crossformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_crossformer, self).__init__(args)
        self.ycat = self.model = None
        self.checkpoint = {}
        self.args = args

        self.use_amp = bool(
            getattr(args, "use_amp", True) and self.device.type == "cuda"
        )
        if self.use_amp:
            self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        else:
            self.scaler = None

    def build_model(self, data, model=None):
        if model is None:
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
        if self.args.optim == "adam":
            model_optim = torch.optim.Adam(
                self.model.parameters(), lr=self.args.learning_rate
            )
        elif self.args.optim == "sgd":
            model_optim = torch.optim.SGD(
                self.model.parameters(), lr=self.args.learning_rate, momentum=0.9
            )
        else:
            model_optim = torch.optim.Adam(
                self.model.parameters(), lr=self.args.learning_rate
            )
        return model_optim

    def _select_criterion(self, ycat, weight):
        def cross_mse_loss_with_nans(input, target):
            assert input.shape[1] == 1
            tv, _ = target
            iv = input
            mi = torch.isnan(iv)
            mask = torch.isnan(tv) | mi
            valid_iv = iv[~mask]
            valid_tv = tv[~mask]

            # Compute MSE
            diff = valid_iv - valid_tv
            mse_loss = diff.pow(2)
            if self.args.over_weight > 0:
                mse_loss = mse_loss + diff.clamp(min=0).pow(2) * self.args.over_weight
            # Compute variance of predictions and targets
            variance_tv = ((valid_tv - valid_tv.mean()).pow(2)).mean()
            variance_iv = ((valid_iv - valid_iv.mean()).pow(2)).mean()

            # Variance loss (maximize variance matching)
            variance_loss = (variance_iv - variance_tv).pow(2) + (
                variance_tv / (1e-8 + variance_iv) - 1
            ).pow(2) * match_lambda(variance_tv, 10)
            loss = torch.log1p(mse_loss.mean() + weight * variance_loss)

            return loss

        def cross_entropy_loss_with_nans(input, target):
            assert input.shape[1] == 1
            _, tc = target

            iv, ic = input[:, :, :-ycat], input[:, 0, -ycat:]
            mc = torch.isnan(ic).any(dim=1)
            if (~mc).sum() == 0:
                return ic.sum() * 0.0

            ic_valid = ic[~mc]
            tc_valid = tc[~mc].long()
            log_probs = F.log_softmax(ic_valid, dim=-1)
            probs = log_probs.exp()
            num_classes = log_probs.size(1)

            device = ic_valid.device
            dtype = ic_valid.dtype

            dist_alpha = getattr(self.args, "dist_alpha", 1.0)
            W = getattr(self, "_class_soft_weights", None)
            if (
                W is None
                or W.size(0) != num_classes
                or W.device != device
                or W.dtype != dtype
            ):
                idx = torch.arange(num_classes, device=device, dtype=dtype)
                dist_mat = (idx.view(-1, 1) - idx.view(1, -1)).abs()
                W = torch.exp(-dist_alpha * dist_mat)
                W = W / W.sum(dim=1, keepdim=True)
                self._class_soft_weights = W

            soft_targets = W.index_select(0, tc_valid)

            ce_per_sample = -(soft_targets * log_probs).sum(dim=1)
            lambda_ce = getattr(self.args, "lambda_ce", 1.0)
            ce_loss = lambda_ce * ce_per_sample.mean()

            use_expected_dist_penalty = getattr(
                self.args, "use_expected_dist_penalty", False
            )
            if use_expected_dist_penalty:
                class_idx = torch.arange(num_classes, device=device, dtype=dtype).view(
                    1, -1
                )
                dist = (class_idx - tc_valid.view(-1, 1)).abs()
                expected_dist = (dist * probs).sum(dim=1)
                lambda_dist = getattr(self.args, "lambda_dist", 0.0)
                dist_loss = lambda_dist * expected_dist.mean()
            else:
                dist_loss = ic_valid.sum() * 0.0

            cls_loss = ce_loss + dist_loss
            mask_penalty = mc.sum() / target[0].numel()

            reg_loss = cross_mse_loss_with_nans(iv, target)
            return cls_loss + mask_penalty + self.args.lambda_mse * reg_loss

        return cross_entropy_loss_with_nans if ycat > 0 else cross_mse_loss_with_nans

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
        version = 251117
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
        # test_data, test_loader = self._get_data(
        #     flag="test", data=data, data_split=data_split
        # )
        self.build_model(train_data)

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.ycat, self.args.weight)
        score = None
        spoch = 0
        if checkpoint is not None:
            try:
                if "model.enc_pos_embedding" in checkpoint[0][0]:
                    self.load_state_dict(checkpoint[0][0])
                else:
                    self.model.load_state_dict(checkpoint[0][0])
                model_optim.load_state_dict(checkpoint[0][1])
                score = abs(checkpoint[1])
                spoch = checkpoint[0][2]
                print_color(
                    93,
                    f"suc to load. score {score:.4g} epoch {spoch} from: {best_model_path}, version {version}",
                )
            except (
                FileNotFoundError,
                RuntimeError,
                IndexError,
                pickle.UnpicklingError,
            ) as e:
                print_color(91, f"version {version} failed to load", e, best_model_path)
        early_stopping = EarlyStopping(
            lradj=self.args.lradj,
            learning_rate=self.args.learning_rate,
            patience=self.args.patience,
            verbose=True,
            best_score=score,
        )
        if getattr(self.args, "profile_mode", False):
            prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            prof.__enter__()

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

                if not torch.isnan(loss).item():
                    train_loss.append(loss.item())
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(model_optim)
                        self.scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                if getattr(self.args, "profile_mode", False):
                    prof.step()
                    if i >= 10:
                        break  # only profile a few batches

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.4g}".format(
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

            if self.args.profile_mode:
                prof.__exit__(None, None, None)
                return

            print(
                "Epoch: {} cost time: {:.4g}".format(
                    epoch + 1, time.time() - epoch_time
                ),
            )
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.4g} "
                f"Vali Loss: {format_nested (vali_loss)} | "
                # f"Test Loss: {format_nested( test_loss)}"
            )

            if early_stopping(
                vali_loss,
                (
                    self.state_dict(),
                    model_optim.state_dict(),
                    epoch + 1,
                    train_data.scaler,
                    train_data.data_split,
                ),
                path,
            ):
                self.test(
                    setting,
                    data,
                    run_metric=True,
                    inverse=True,
                    test_data=(vali_data, vali_loader),
                )
            if early_stopping.early_stop:
                print_color(95, "Early stopping")
                break

            if early_stopping.adjust_learning_rate(model_optim):
                best_model_path = path + "/" + "checkpoint.pth"
                checkpoint = list(torch.load(best_model_path, weights_only=False))
                if "model.enc_pos_embedding" in checkpoint[0][0]:
                    self.load_state_dict(checkpoint[0][0])
                else:
                    self.model.load_state_dict(checkpoint[0][0])
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
        test_data=None,
    ):
        if test_data is None:
            key = setting + data
            if key not in self.checkpoint:
                best_model_path = (
                    os.path.join(self.args.checkpoints, key) + "/crossformer.pkl"
                )
                try:
                    self.checkpoint[key] = torch.load(
                        best_model_path, weights_only=False
                    )
                    print_color(93, "suc to load", best_model_path)
                except (
                    FileNotFoundError,
                    RuntimeError,
                    IndexError,
                    pickle.UnpicklingError,
                ) as e:
                    print_color(91, "failed to load", e, best_model_path)
            test_data, test_loader = self._get_data(
                data=data,
                flag="test",
                scaler=self.checkpoint[key][1],
                data_path=data_path,
                data_split=(
                    self.checkpoint[key][2] if len(self.checkpoint[key]) > 2 else None
                ),
            )
            self.build_model(test_data, model=self.checkpoint[key][0])
        else:
            test_data, test_loader = test_data

        self.model.eval()

        preds = []
        trues = []
        metrics = None

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(test_data, batch_x, batch_y)

                if inverse:
                    pred, true = self._inverse(test_data, pred, true)

                if save_pred or run_metric:
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

        if preds and trues:
            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
            if run_metric:
                metric_fn = make_metric(self.ycat)
                metrics = metric_fn(preds, trues)
                mae, mse, rmse, mape, mspe, accr = metrics
                print_color(
                    93,
                    f"mae:{mae:.3f}, mse:{mse:.3f}, rmse:{rmse:.3f}, mape:{mape:.3f}, {'mspe' if accr==-1 else 'mdst'}:{mspe:.3f}, accr:{accr}",
                )

        return preds, trues, metrics

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse=False):
        batch_x = [x.float().to(self.device) for x in batch_x]
        batch_y = batch_y[0].float().to(self.device), batch_y[1].type(
            torch.LongTensor
        ).to(self.device)

        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type):
                outputs = self.model(batch_x)
        else:
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
