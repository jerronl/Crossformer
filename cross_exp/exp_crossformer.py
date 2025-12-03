import os
import time
import pickle

import warnings
import numpy as np

import torch.profiler
import torch
from torch import nn
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


class HybridLoss(nn.Module):
    def __init__(self, args, ycat):
        super().__init__()
        self.args = args
        self.ycat = ycat
        self.register_buffer("class_weights", None)

    def _ensure_class_weights(self, device, dtype):
        if self.ycat <= 0:
            return None

        if self.class_weights is None:
            idx = torch.arange(self.ycat, device=device, dtype=dtype)
            dist_mat = (idx[:, None] - idx[None, :]).abs()

            if getattr(self.args, "use_gaussian_class_weights", False):
                max_diff = max(self.ycat - 1, 1)
                dist_norm = dist_mat / max_diff
                gamma = getattr(self.args, "dist_alpha", 3.0)
                W = torch.exp(-gamma * dist_norm * dist_norm)
            else:
                alpha = getattr(self.args, "dist_alpha", 3.0)
                W = torch.exp(-alpha * dist_mat)

            W = W / W.sum(dim=1, keepdim=True)
            self.class_weights = W
        else:
            if (self.class_weights.device != device) or (
                self.class_weights.dtype != dtype
            ):
                self.class_weights = self.class_weights.to(device=device, dtype=dtype)

        return self.class_weights

    def regression_loss(self, pred, true):
        pred = pred.squeeze(1)
        true = true.squeeze(1)
        tnum = true.numel()

        mask = torch.isnan(pred) | torch.isnan(true)
        pred = pred[~mask]
        true = true[~mask]

        gain = getattr(self.args, "gain_reg", 1.0)
        error = gain * (pred - true)

        over = torch.relu(error)
        under = torch.relu(-error)
        w_over = getattr(self.args, "weight_over", 3.0)
        w_under = getattr(self.args, "weight_under", 1.0)
        asym = w_over * over**2 + w_under * under**2
        asym_loss = asym.mean()

        abs_err = error.abs()
        if abs_err.numel() > 0:
            median = abs_err.median()
        else:
            median = torch.tensor(0.0, device=abs_err.device, dtype=abs_err.dtype)

        tail_k = getattr(self.args, "tail_k", 3.0)
        tail_th = tail_k * (median + 1e-8)
        tail = torch.relu(abs_err - tail_th)
        lambda_tail = getattr(self.args, "lambda_tail", 0.0)
        tail_loss = lambda_tail * (tail**2).mean()

        lambda_anti = getattr(self.args, "lambda_anti", 0.0)
        anti_term = torch.relu(-error * true)
        anti_loss = lambda_anti * anti_term.mean()

        q = getattr(self.args, "quantile", 0.5)
        lambda_q = getattr(self.args, "lambda_quantile", 0.0)
        e = error
        q_loss = torch.max(q * e, (q - 1.0) * e).mean()
        quantile_loss = lambda_q * q_loss

        return (
            asym_loss
            + tail_loss
            + anti_loss
            + quantile_loss
            + mask.sum() / tnum * 100.0
        )

    def classification_loss(self, logits, tc):
        logits = logits.squeeze(1)
        tc = tc.squeeze(-1)
        tnum = tc.numel()

        mask = torch.isnan(logits).any(dim=1)

        logits = logits[~mask]
        tc = tc[~mask].long()

        if logits.numel() == 0:
            return logits.sum() * 0.0

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        W = self._ensure_class_weights(logits.device, logits.dtype)
        if W is not None:
            soft_targets = W.index_select(0, tc)
        else:
            soft_targets = F.one_hot(tc, num_classes=logits.size(-1)).to(logits.dtype)

        lambda_ce = getattr(self.args, "lambda_ce", 2.0)
        ce = -(soft_targets * log_probs).sum(dim=1).mean() * lambda_ce

        dist_loss = logits.sum() * 0.0
        if getattr(self.args, "use_expected_dist_penalty", False):
            num_classes = logits.size(-1)
            idx = torch.arange(num_classes, device=logits.device, dtype=logits.dtype)
            dist = (idx[None, :] - tc[:, None]).abs()

            if getattr(self.args, "use_normalized_expected_dist", False):
                max_diff = max(num_classes - 1, 1)
                dist = dist / max_diff

            expected_dist = (dist * probs).sum(dim=1)
            lambda_dist = getattr(self.args, "lambda_dist", 0.0)
            dist_loss = lambda_dist * expected_dist.mean()

        lambda_entropy = getattr(self.args, "lambda_entropy", 0.01)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        entropy_loss = lambda_entropy * entropy

        lambda_smooth = getattr(self.args, "lambda_smooth", 0.0)
        smooth_loss = logits.sum() * 0.0
        if lambda_smooth > 0.0:
            diff = probs[:, 1:] - probs[:, :-1]
            smooth_loss = lambda_smooth * (diff * diff).mean()

        lambda_mode = getattr(self.args, "lambda_mode", 0.0)
        mode_loss = logits.sum() * 0.0
        if lambda_mode > 0.0:
            mode = probs.argmax(dim=-1)
            mode_loss = lambda_mode * (mode.float() - tc.float()).abs().mean()

        lambda_emd = getattr(self.args, "lambda_emd", 0.0)
        emd_loss = logits.sum() * 0.0
        if lambda_emd > 0.0:
            num_classes = probs.size(-1)
            cdf_pred = probs.cumsum(dim=-1)
            onehot = F.one_hot(tc, num_classes=num_classes).to(probs.dtype)
            cdf_true = onehot.cumsum(dim=-1)
            emd_loss = lambda_emd * (cdf_pred - cdf_true).abs().mean()

        lambda_sep = getattr(self.args, "lambda_sep", 0.0)
        sep_loss = logits.sum() * 0.0
        if lambda_sep > 0.0:
            classes = tc.unique()
            means = []
            for c in classes:
                mask_c = tc == c
                if mask_c.sum() < 2:
                    continue
                means.append(probs[mask_c].mean(dim=0, keepdim=True))
            if len(means) >= 2:
                means = torch.cat(means, dim=0)
                diffs = []
                n = means.size(0)
                for i in range(n):
                    for j in range(i + 1, n):
                        # 核心修改：在平方和内部增加 1e-8 防止数值溢出或梯度为 NaN
                        d = ((means[i] - means[j]).pow(2).sum() + 1e-8).sqrt()
                        diffs.append(d)
                if diffs:
                    sep = torch.stack(diffs).mean()
                    sep_loss = -lambda_sep * sep

        return (
            ce
            + dist_loss
            + entropy_loss
            + smooth_loss
            + mode_loss
            + emd_loss
            + sep_loss
            + mask.sum() / tnum * 100.0
        )

    def forward(self, input, target):
        tv, tc = target
        if self.ycat == 0:
            return self.regression_loss(input, tv)

        iv = input[..., : -self.ycat]
        ic = input[..., -self.ycat :]

        reg = self.regression_loss(iv, tv)
        cls = self.classification_loss(ic, tc)
        lambda_mse = getattr(self.args, "lambda_mse", 1.0)
        return cls + lambda_mse * reg


class Exp_crossformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_crossformer, self).__init__(args)
        self.ycat = self.model = None
        self.current_model_key = None
        self.checkpoint = {}
        self.args = args
        self.epochs = -1

        self.use_amp = bool(
            getattr(args, "use_amp", False) and self.device.type == "cuda"
        )
        if self.use_amp:
            self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        else:
            self.scaler = None

    def get_epochs(self):
        return self.epochs

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

    def _select_criterion(self):
        return HybridLoss(self.args, self.ycat)

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

    def _load_checkpoint_file(self, path):
        try:
            checkpoint = torch.load(path, weights_only=False)
        except (
            FileNotFoundError,
            RuntimeError,
            IndexError,
            pickle.UnpicklingError,
            ValueError,
        ) as e:
            return None

        state = {}
        if isinstance(checkpoint, list):
            inner = checkpoint[0]
            state["model_state"] = inner[0]
            state["optimizer_state"] = inner[1]
            state["epoch"] = inner[2]
            state["scaler"] = inner[3]
            state["data_split"] = inner[4]
            state["score"] = checkpoint[1]
        elif isinstance(checkpoint, dict):
            state = checkpoint
        else:
            raise ValueError(f"Unknown checkpoint format at {path}")
        return state

    def _apply_checkpoint(self, state, model_optim=None, version=None, path=None):
        if state is None:
            return 0, np.inf

        try:
            state_dict = state["model_state"]

            if "model.enc_pos_embedding" in state_dict:
                self.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)

            if model_optim is not None and "optimizer_state" in state:
                model_optim.load_state_dict(state["optimizer_state"])

            score = abs(state["score"])
            epoch = state["epoch"]
            return epoch, score

        except Exception as e:
            print_color(
                91, f"version {version} failed to load state dict from {path}", e
            )
            return 0, np.inf

    def train(self, setting, data):
        version = 25112112
        checkpoint_state = None
        data_split = None
        key = setting + data
        path = os.path.join(self.args.checkpoints, key)

        if not os.path.exists(path):
            os.makedirs(path)
        best_model_path = path + "/" + "checkpoint.pth"

        if self.args.resume:
            checkpoint_state = self._load_checkpoint_file(best_model_path)
            if checkpoint_state is None:
                print_color(
                    91,
                    "failed to load checkpoint during resume. Starting from scratch.",
                    best_model_path,
                )
            else:
                data_split = checkpoint_state.get("data_split")

        train_data, train_loader = self._get_data(
            flag="train", data=data, data_split=data_split
        )
        vali_data, vali_loader = self._get_data(
            flag="val", data=data, data_split=train_data.data_split
        )

        self.build_model(train_data)
        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        score = np.inf
        start_epoch = 0

        if checkpoint_state is not None:
            start_epoch, score_loaded = self._apply_checkpoint(
                checkpoint_state,
                model_optim=model_optim,
                version=version,
                path=best_model_path,
            )
            score = score_loaded

            if score != np.inf:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
            else:
                vali_loss = np.inf

            score = vali_loss
            print_color(
                93,
                f"suc to load. score {score:.4g} vali {vali_loss:.4g} epoch {start_epoch} from: {best_model_path}, version {version}",
            )

        if start_epoch < 1:
            init_state = {
                "model_state": self.model.state_dict(),
                "optimizer_state": model_optim.state_dict(),
                "epoch": start_epoch,
                "scaler": train_data.scaler,
                "data_split": train_data.data_split,
                "score": np.inf,
            }
            torch.save(init_state, best_model_path)
            print_color(94, f"init model saved to:", best_model_path,train_data.data_split)

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

        for epoch in range(start_epoch, self.args.train_epochs):
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
                        self.scaler.unscale_(model_optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(model_optim)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        model_optim.step()

                if getattr(self.args, "profile_mode", False):
                    prof.step()
                    if i >= 10:
                        break

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

            if getattr(self.args, "profile_mode", False):
                prof.__exit__(None, None, None)
                return

            print(
                "Epoch: {} cost time: {:.4g}".format(
                    epoch + 1, time.time() - epoch_time
                )
            )
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.4g} "
                f"Vali Loss: {format_nested(vali_loss)}"
            )

            current_state = {
                "model_state": self.model.state_dict(),
                "optimizer_state": model_optim.state_dict(),
                "epoch": epoch + 1,
                "scaler": train_data.scaler,
                "data_split": train_data.data_split,
                "score": vali_loss,
            }

            if early_stopping(vali_loss, current_state, path) and getattr(
                self.args, "metric_in_train", False
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
                best_state = self._load_checkpoint_file(best_model_path)
                self._apply_checkpoint(
                    best_state,
                    model_optim=model_optim,
                    version=version,
                    path=best_model_path,
                )
            self.epochs = epoch

        best_state = self._load_checkpoint_file(best_model_path)
        self._apply_checkpoint(
            best_state, model_optim=None, version=version, path=best_model_path
        )

        if best_state is None:
            print_color(
                91, "Checkpoint loading failed, using final model state for saving."
            )
            best_state = {
                "model_state": self.model.state_dict(),
                "optimizer_state": {},
                "epoch": self.epochs + 1,
                "scaler": train_data.scaler,
                "data_split": train_data.data_split,
                "score": vali_loss if "vali_loss" in locals() else np.inf,
            }
        else:
            best_state["model_state"] = self.model.state_dict()
            best_state["data_split"] = train_data.data_split
        torch.save(best_state, path + "/checkpoint.pth")

        self.checkpoint[key] = (
            self.model,
            best_state["scaler"],
            best_state["data_split"],
            best_state["epoch"],
        )
        torch.save(self.checkpoint[key], path + "/crossformer.pkl")
        self.current_model_key = key

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

        if test_data is None:
            scaler = (
                self.checkpoint[key][1]
                if key in self.checkpoint and len(self.checkpoint[key]) > 1
                else None
            )
            data_split = (
                self.checkpoint[key][2]
                if key in self.checkpoint and len(self.checkpoint[key]) > 2
                else None
            )

            test_data, test_loader = self._get_data(
                data=data,
                flag="test",
                scaler=scaler,
                data_path=data_path,
                data_split=data_split,
            )
        else:
            test_data, test_loader = test_data

        if self.model is None or self.current_model_key != key:
            if key in self.checkpoint:
                self.build_model(test_data, model=self.checkpoint[key][0])
                self.current_model_key = key
            else:
                raise RuntimeError(
                    f"Exp_crossformer.test() called with key='{key}', "
                    f"but no model is available in checkpoint and self.model is None. "
                    f"Please ensure train() has been run or checkpoint exists."
                )

        if self.checkpoint.get(key) is not None:
            if len(self.checkpoint[key]) > 3:
                self.epochs = self.checkpoint[key][3]

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
                metric_fn, metric_names = make_metric(self.ycat)
                metrics = metric_fn(preds, trues)
                print_color(
                    93,
                    ", ".join(
                        f"{name}:{value:.4g}"
                        for name, value in zip(metric_names, metrics)
                    ),
                )

        return preds, trues, metrics

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse=False):
        batch_x = [x.float().to(self.device) for x in batch_x]
        batch_y = batch_y[0].float().to(self.device), batch_y[1].type(
            torch.LongTensor
        ).to(self.device)

        if self.use_amp:
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                outputs = self.model(batch_x)
        else:
            outputs = self.model(batch_x)
        if inverse:
            return self._inverse(dataset_object, outputs, batch_y)
        return outputs, batch_y

    def _inverse(self, dataset_object, outputs, batch_y):
        if isinstance(outputs, torch.Tensor) and outputs.dtype == torch.bfloat16:
            outputs = outputs.to(torch.float32)
        if isinstance(batch_y, torch.Tensor) and batch_y.dtype == torch.bfloat16:
            batch_y = batch_y.to(torch.float32)
        elif isinstance(batch_y, tuple):
            batch_y = tuple(
                (
                    t.to(torch.float32)
                    if isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16
                    else t
                )
                for t in batch_y
            )

        if dataset_object.ycat > 0:
            outputs[:, :, -dataset_object.ycat :] = F.softmax(
                outputs[:, :, -dataset_object.ycat :], dim=2
            )

        outputs = dataset_object.inverse_transform(outputs)
        batch_y = dataset_object.inverse_transform(batch_y)
        return outputs, batch_y
