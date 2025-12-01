"""dataset MTS."""

import warnings
import os
import calendar
from random import uniform
from datetime import datetime, timedelta
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import torch.nn.functional as F

# from einops import rearrange
from torch.utils.data import Dataset
from data.data_def import data_columns, data_names

warnings.filterwarnings("ignore")

# Todo: fix this bug
epoch0 = datetime(1899, 12, 30)
# base_date = datetime(1899, 12, 30)


def datetime_to_excel_serial(dt):
    delta = dt - epoch0
    return delta.days + delta.seconds / 86400  # (24 * 60 * 60)


def excel_date(x):
    return datetime_to_excel_serial(
        datetime(
            int(x[1:5]),
            int(x[6:8]),
            int(x[9:11]),
            int(x[12:14]),
            int(x[15:17]),
            int(x[18:20]),
        )
    )


def cyclic_encode(original, period):
    return [
        np.sin(2 * np.pi * original.astype(float) / period),
        np.cos(2 * np.pi * original.astype(float) / period),
    ]


def cyclic_t(x):
    if len(x) < 1:
        return [pd.DataFrame()]
    if np.issubdtype(x.dtype, float):
        t = np.vectorize(lambda x: (epoch0 + timedelta(days=x)).timetuple())(x)
    elif np.issubdtype(x.dtype, datetime):
        t = np.vectorize(lambda x: x.timetuple())(x)
    else:
        raise TypeError(f"unsuported type {x.dtype}")
    tm_yday = cyclic_encode(
        t[7] - 1, np.vectorize(lambda x: 365 - 28 + calendar.monthrange(x, 2)[1])(t[0])
    )
    tm_mday = cyclic_encode(
        t[2] - 1, np.vectorize(lambda x, y: calendar.monthrange(x, y)[1])(t[0], t[1])
    )
    tm_wday = cyclic_encode(t[6], 7)
    return tm_yday + tm_mday + tm_wday


class MixedStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, m):
        self.m = m
        self.scaler_front = StandardScaler()
        self.scaler_rest = StandardScaler()

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.orig_shape = X.shape
        self.n_features_in_ = X.shape[-1]
        if self.m > self.n_features_in_:
            raise ValueError(
                f"m = {self.m} exceeds number of features = {self.n_features_in_}"
            )

        # 前 m 列分别标准化
        X_front = X[..., : self.m].reshape(-1, self.m)
        self.scaler_front.fit(X_front)

        # 后 k-m 列整体 flatten 成一列
        X_rest = X[..., self.m :].reshape(-1, 1)
        self.scaler_rest.fit(X_rest)

        # 保存参数
        self.front_mean_ = self.scaler_front.mean_
        self.front_scale_ = self.scaler_front.scale_
        self.rest_mean_ = self.scaler_rest.mean_[0]
        self.rest_scale_ = self.scaler_rest.scale_[0]
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.shape[-1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[-1]}."
            )

        orig_shape = X.shape

        # 标准化前 m 列
        X_front = X[..., : self.m].reshape(-1, self.m)
        X_front_scaled = self.scaler_front.transform(X_front).reshape(
            *orig_shape[:-1], self.m
        )

        # 后面 flatten 标准化
        X_rest = X[..., self.m :].reshape(-1, 1)
        X_rest_scaled = self.scaler_rest.transform(X_rest).reshape(
            *orig_shape[:-1], self.n_features_in_ - self.m
        )

        return np.concatenate([X_front_scaled, X_rest_scaled], axis=-1)

    def inverse_transform(self, X_scaled):
        X_scaled = np.asarray(X_scaled)
        if X_scaled.shape[-1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X_scaled.shape[-1]}."
            )

        orig_shape = X_scaled.shape

        X_front_scaled = X_scaled[..., : self.m].reshape(-1, self.m)
        X_front_orig = self.scaler_front.inverse_transform(X_front_scaled).reshape(
            *orig_shape[:-1], self.m
        )

        X_rest_scaled = X_scaled[..., self.m :].reshape(-1, 1)
        X_rest_orig = self.scaler_rest.inverse_transform(X_rest_scaled).reshape(
            *orig_shape[:-1], self.n_features_in_ - self.m
        )

        return np.concatenate([X_front_orig, X_rest_orig], axis=-1)


class DatasetMTS(Dataset):
    datas = {}

    @classmethod
    def clear(cls):
        cls.datas = {}

    def __init__(
        self,
        root_path,
        data_path,
        data_name,
        in_len,
        flag="train",
        data_split=None,
        query=False,
        scaler=None,
    ):
        if data_split is None:
            data_split = [0.7, 0.1, 0.2]
        type_map = {"train": 0, "val": 1, "test": 2}
        assert flag in type_map
        self.set_type = type_map[flag]
        # info
        self.in_len = in_len
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.data_name = data_name
        self.query = query
        self.scaler = scaler
        self.__read_data__()
        self.shape = self.data[2][self.set_type][0]
        self.scaler = self.data[0]
        self.data_dim, _, self.out_dim, self.ycat, self.sect, self.sp = self.data[1]

    def _read_files_and_split(self, cols, dtm0):
        df_raws = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
        if isinstance(self.data_path, pd.DataFrame):
            df_raws[self.set_type] = self.data_path
        else:
            if isinstance(self.data_split, dict):
                data_split = np.array([0, 0.7, 0.85, 1])
            else:
                data_split = np.cumsum([0] + self.data_split)
                self.data_split = {}

            for table in self.data_path:
                df = pd.read_csv(os.path.join(self.root_path, table)).replace(
                    -9999900, float("nan")
                )
                df = df[~df[dtm0].isna()]
                if not cols["xvsp"] and "spot" in df.columns:
                    df.drop(columns="spot", inplace=True)
                if self.query is not None or not cols["xvsp"]:
                    df["day"] = pd.to_datetime(df["date"].str.replace("#", "")).dt.date
                    df["horizon"] = df[f"{dtm0}_{self.in_len}"] - df[dtm0]
                    lasttime = df.groupby(["day"])["date"].max().values
                    df = df[(df["date"].isin(lasttime)) & (df["horizon"] > 0)]

                ds = None
                if self.query is not None:
                    is_oos_mode = "cutdate" in self.query or isinstance(
                        self.data_path, pd.DataFrame
                    )

                    df = df.query(self.query)

                    if is_oos_mode:
                        ds = [0, 0, 0, len(df) - 1]

                if ds is None:
                    if table in self.data_split and self.data_split[table][-1] < len(
                        df
                    ):
                        ds = self.data_split[table]
                    else:
                        ds = (data_split * uniform(0.8, 0.9) * len(df)).astype(int)
                        self.data_split[table] = ds

                print(table, df["date"].iloc[ds])
                df["date"] = np.vectorize(excel_date)(df["date"])
                for i, df_raw in enumerate(df_raws):
                    df_raws[i] = pd.concat(
                        [
                            df_raw,
                            df.iloc[ds[i] : ds[i + 1]],
                        ]
                    )
                i = 0 if ds[1] else self.set_type
                df_raws[i] = pd.concat([df_raws[i], df.iloc[ds[-1] :]])
        return df_raws

    def _extract_initial_arrays(self, df_raws, cols, names):
        vy, vnp, vnpt, vpc, vvs, vpct, vsp, vspt = names
        xnp, xpc, xvsp, xvs, y, cyclics, xsp = [[] for _ in range(7)]
        for i, df_raw in enumerate(df_raws):
            if len(df_raw.columns) < 1:
                df_raw = pd.DataFrame(columns=df_raws[self.set_type].columns)
                df_raws[i] = df_raw
            xnp.append(df_raw[vnpt].values.astype(float).reshape((-1, len(vnp))))
            xsp.append(df_raw[vspt].values.astype(float).reshape((-1, len(vsp))))
            xpc.append(df_raw[vpct].values.reshape((-1, 1)))
            _y = df_raw[vy].values
            if cols["xvsp"]:
                xvsp.append(df_raw["spot"].values.reshape(-1, 1))
                y.append([_y, 0])
            else:
                xvsp.append(np.zeros((len(df_raw), 1), float))
                _y = np.column_stack((_y, abs(_y[:, 0] - cols["ycat"] / 2.0)))
                y.append([_y, _y[:, 0] - 1])

            assert (
                len(xnp[-1]) < 1
                or np.isnan(xnp[-1]).sum() == 0
                and np.isnan(xpc[-1]).sum() == 0
            )
        return xnp, xsp, xpc, xvsp, xvs, y, cyclics

    def _prepare_scalers(self, xs, y, cols):
        y1 = len(cols["vml"])
        if self.scaler is None:
            scalers = [StandardScaler() for _ in xs] + [MixedStandardScaler(y1)]
            for i, x in enumerate(xs):
                scalers[i].fit(x[0])
            scalers[i + 1].fit(y[0][0])
        else:
            scalers = self.scaler
        return scalers

    def _apply_transforms(self, df_raws, scalers, arrays, cols, names):
        xnp, xsp, xpc, xvsp, xvs, y, cyclics = arrays
        vy, vnp, vnpt, vpc, vvs, vpct, vsp, vspt = names
        xs = [xnp, xsp, xpc, xvsp]
        ivs = [vnp.index(v) for v in vvs]
        idat = [vnp.index(v) for v in ["date"]]
        idatv = [vvs.index(v) for v in ["date"]]
        icyc = [[vnp.index(v) for v, c in cols["cyc"]], [c for v, c in cols["cyc"]]]

        for i, df_raw in enumerate(df_raws):
            if len(xnp[i]) < 1:
                cyclics.append([])
                xvs.append([])
            else:
                c = np.concatenate(
                    cyclic_t(xnp[i][:, idat])
                    + cyclic_encode(xnp[i][:, icyc[0]], icyc[1]),
                    axis=1,
                )
                xx, xsp[i], xpc[i] = [
                    scalers[j].transform(x[i]).reshape((-1, self.in_len, x[i].shape[1]))
                    for j, x in enumerate(xs[:3])
                ]
                cyclics.append(
                    np.concatenate(
                        [
                            xx[:, :, : len(ivs)],
                            c.reshape((-1, self.in_len, c.shape[1])),
                        ],
                        axis=2,
                    )
                )
                xnp[i] = xx[:, :, len(ivs) :]
                x = df_raw[vvs].values
                c = np.concatenate(
                    cyclic_t(x[:, idatv]) + cyclic_encode(x[:, icyc[0]], icyc[1]),
                    axis=1,
                )
                x = (x - scalers[0].mean_[ivs]) / scalers[0].scale_[ivs]
                xvs.append(np.concatenate([x, c], axis=1))
                xvsp[i] = scalers[-2].transform(xvsp[i])
                y[i][0] = (
                    scalers[-1].transform(y[i][0]).reshape(-1, 1, y[i][0].shape[1])
                )
                assert (
                    np.isnan(xnp[i]).sum() == 0
                    and np.isnan(xsp[i]).sum() == 0
                    and np.isnan(xpc[i]).sum() == 0
                    and np.isnan(cyclics[-1]).sum() == 0
                    and np.isnan(xvs[-1]).sum() == 0
                    and np.isnan(xvsp[-1]).sum() == 0
                )
        return xnp, xsp, xpc, xvsp, xvs, y, cyclics

    def __read_data__(self):
        cols = data_columns(self.data_name)
        if isinstance(self.data_path, pd.DataFrame):
            df_raws = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
            df_raws[self.set_type] = self.data_path
            cache_key = None
        else:
            if isinstance(self.data_path, list):
                cache_key = tuple(self.data_path), self.data_name
            else:
                cache_key = self.data_path, self.data_name

            if cache_key in self.__class__.datas:
                self.data = self.__class__.datas[cache_key]
                return

            df_raws = self._read_files_and_split(cols, cols["dtm0"])

        names = data_names(cols, self.in_len)
        vy, vnp, vnpt, vpc, vvs, vpct, vsp, vspt = names

        arrays = self._extract_initial_arrays(df_raws, cols, names)
        xnp, xsp, xpc, xvsp, xvs, y, cyclics = arrays

        xs = [xnp, xsp, xpc, xvsp]
        scalers = self._prepare_scalers(xs, y, cols)

        arrays = self._apply_transforms(df_raws, scalers, arrays, cols, names)
        xnp, xsp, xpc, xvsp, xvs, y, cyclics = arrays

        assert len(y) >= 3, "Expected at least 3 data splits (train/val/test)"

        self.data = [
            scalers,
            (
                xnp[self.set_type].shape[2]
                + cyclics[self.set_type].shape[2]
                + xpc[self.set_type].shape[2],
                xvs[self.set_type].shape[1] + xvsp[self.set_type].shape[1],
                cols["ycat"] + y[2][0].shape[2],
                cols["ycat"],
                cols["sect"],
                len(vsp) // cols["sect"],
            ),
            list(
                zip(
                    [x.shape for x in xnp],
                    xnp,
                    xsp,
                    cyclics,
                    xpc,
                    xvs,
                    xvsp,
                    y,
                )
            ),
        ]
        if cache_key is not None:
            self.__class__.datas[cache_key] = self.data

    def __getitem__(self, index):
        (
            _,
            xnp,
            xsp,
            cyclic,
            xpc,
            xvs,
            xvsp,
            y,
        ) = self.data[
            2
        ][self.set_type]

        seq_x = (
            xnp[index],
            xsp[index],
            cyclic[index],
            xpc[index],
            xvs[index],
            xvsp[index],
        )
        seq_y = y[0][index], y[1][index] if isinstance(y[1], np.ndarray) else 0

        return seq_x, seq_y

    def __len__(self):
        return self.shape[0]

    def inverse_transform(self, data):
        dt = (
            data[0].detach().cpu()
            if isinstance(data, (tuple, list))
            else data.detach().cpu()
        )
        w = self.scaler[-1].n_features_in_
        dt = np.concatenate(
            (self.scaler[-1].inverse_transform(dt[:, 0, :w]), dt[:, 0, w:]), axis=1
        )
        return (dt, data[1].detach().cpu()) if isinstance(data, (tuple, list)) else dt
