import os, sys, traceback
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

        k = 10
        idx = torch.arange(self.ycat)  # (C,)
        dist = (idx.view(-1, 1) - idx.view(1, -1)).abs()  # (C, C)  距离=索引差的绝对值
        far_idx_lut = torch.topk(
            dist, k=k, dim=1, largest=True, sorted=False
        ).indices  # (C, K)
        self.register_buffer(
            "far_idx_lut", far_idx_lut.to(self.device), persistent=True
        )

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
        model_optim = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, ycat, weight):
        # cel = nn.CrossEntropyLoss()
        def cross_entropy_mse_loss_with_nans(input, target):
            assert input.shape[1] == 1
            _, tc = target
            iv, ic = input[:, :, :-ycat], input[:, 0, -ycat:]

            mc = torch.isnan(ic).any(dim=1)
            if (~mc).sum() == 0:
                return ic.sum() * 0.0

            log_probs = F.log_softmax(ic[~mc], dim=-1)
            probs = log_probs.exp()
            tc_valid = tc[~mc].long()
            num_classes = log_probs.size(1)

            left = torch.clamp(tc_valid - 1, min=0)
            right = torch.clamp(tc_valid + 1, max=num_classes - 1)
            candidates = torch.stack([tc_valid, left, right], dim=1)

            weights = torch.tensor(
                [1.0, self.args.ajc_weight, self.args.ajc_weight],
                device=tc_valid.device,
                dtype=log_probs.dtype,
            ).view(1, 3)

            try:
                ce_adj = -(log_probs.gather(1, candidates) * weights).mean() * 3.0
            except Exception as e:
                tb = sys.exc_info()[2]
                while tb.tb_next:
                    tb = tb.tb_next
                frame = tb.tb_frame
                print("Exception at:", frame.f_code.co_filename, "line", frame.f_lineno)
                print("Locals in this frame:")
                for k, v in frame.f_locals.items():
                    try:
                        print(f"  {k}: {type(v)} {getattr(v, 'shape', '')}")
                    except Exception:
                        print(f"  {k}: {v}")
                traceback.print_exc()
                raise

            far_idx = self.far_idx_lut[tc_valid]  # (Bv, K)
            far_prob_sum = probs.gather(1, far_idx).sum(dim=1)  # (Bv,)
            far_loss = far_prob_sum.mean()

            cross_entropy_loss = ce_adj + self.args.ajc_weight * far_loss * 4

            # MSE (can replace with asymmetric one if desired)
            mse_loss = cross_mse_loss_with_nans(iv, target)

            return cross_entropy_loss + mse_loss + mc.sum() / target[0].numel()

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

            return (
                (mse_loss.mean() + weight * variance_loss) ** 0.5
            ) * 10 + mi.sum() / target[0].numel()

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
                    f"suc to load. score {score:.4g} epoch {spoch} from:",
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
        if self.args.profile_mode:
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
                if ~torch.isnan(loss):
                    train_loss.append(loss.item())
                    loss.backward()
                    model_optim.step()

                if self.args.profile_mode:
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
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.4g} "
                f"Vali Loss: {format_nested (vali_loss)} | "
                f"Test Loss: {format_nested( test_loss)}"
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

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(test_data, batch_x, batch_y)
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
