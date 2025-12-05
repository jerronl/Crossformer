import warnings
import os
import calendar
from random import uniform
from datetime import datetime, timedelta
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import torch.nn.functional as F
from torch.utils.data import Dataset
from data.data_def import data_columns, data_names
from utils.tools import print_color

warnings.filterwarnings("ignore")
epoch0 = datetime(1899, 12, 30)


def datetime_to_excel_serial(dt):
    delta = dt - epoch0
    return delta.days + delta.seconds / 86400.0  # (24 * 60 * 60)


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
    original = np.asanyarray(original)
    period = np.asanyarray(period)

    val = 2 * np.pi * original.astype(float) / period
    return [
        np.sin(val).reshape(-1, 1),
        np.cos(val).reshape(-1, 1),
    ]


def cyclic_t(x):
    if len(x) < 1:
        return [pd.DataFrame()]

    x = np.asarray(x).flatten()

    if np.issubdtype(x.dtype, np.number):
        dt_series = pd.Series(pd.to_datetime(x, unit="D", origin=epoch0))
    else:
        dt_series = pd.Series(pd.to_datetime(x))

    days_in_year = np.where(dt_series.dt.is_leap_year, 366.0, 365.0)
    tm_yday = cyclic_encode(dt_series.dt.dayofyear - 1, days_in_year)

    days_in_month = dt_series.dt.days_in_month.values
    tm_mday = cyclic_encode(dt_series.dt.day - 1, days_in_month)

    tm_wday = cyclic_encode(dt_series.dt.dayofweek.values, 7)
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
        X_front = X[..., : self.m].reshape(-1, self.m)
        self.scaler_front.fit(X_front)
        X_rest = X[..., self.m :].reshape(-1, 1)
        self.scaler_rest.fit(X_rest)
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
        X_front = X[..., : self.m].reshape(-1, self.m)
        X_front_scaled = self.scaler_front.transform(X_front).reshape(
            *orig_shape[:-1], self.m
        )
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

    def _get_files_signature(self):
        if isinstance(self.data_path, pd.DataFrame):
            return "dataframe_input"
        file_list = (
            self.data_path if isinstance(self.data_path, list) else [self.data_path]
        )
        sig = []
        for file_name in file_list:
            full_path = os.path.join(self.root_path, file_name)
            if os.path.exists(full_path):
                stat = os.stat(full_path)
                sig.append((file_name, stat.st_size, stat.st_mtime))
            else:
                sig.append((file_name, "missing"))
        return tuple(sig)

    def _get_split_signature(self):
        if isinstance(self.data_split, list):
            return ("list", tuple(self.data_split))
        elif isinstance(self.data_split, dict):
            items = []
            for k, v in self.data_split.items():
                if isinstance(v, np.ndarray):
                    v = tuple(v.tolist())
                items.append((k, v))
            return ("dict", tuple(sorted(items)))
        return ("other", str(self.data_split))

    def _get_cache_key(self):
        return (
            self._get_files_signature(),
            self.data_name,
            self._get_split_signature(),
            str(self.query),
        )

    def _check_cache_and_load(self):
        if isinstance(self.data_path, pd.DataFrame):
            return False, None
        file_sig = self._get_files_signature()
        split_sig = self._get_split_signature()
        cache_key = (file_sig, self.data_name, split_sig, str(self.query))
        if cache_key in self.__class__.datas:
            cached_value = self.__class__.datas[cache_key]
            if split_sig[0] == "list":
                cache_key = (file_sig, self.data_name, cached_value, str(self.query))
                cached_value = self.__class__.datas[cache_key]
            self.data, self.data_split = cached_value
            return True, cache_key
        if split_sig[0] == "list":
            target_ratios = list(split_sig[1])
            total_ratio = sum(target_ratios)
            if total_ratio <= 0:
                return False, cache_key
            norm_target = [r / total_ratio for r in target_ratios]
            for existing_key, cached_data in self.__class__.datas.items():
                ex_file_sig, ex_data_name, ex_split_sig, ex_query = existing_key
                if ex_data_name != self.data_name or ex_file_sig != file_sig:
                    continue
                if ex_query != str(self.query):
                    continue
                if ex_split_sig[0] != "dict":
                    continue
                try:
                    cached_lengths = [d[1].shape[0] for d in cached_data[2]]
                    total_len = sum(cached_lengths)
                    if total_len == 0:
                        continue
                    cached_ratios = [l / total_len for l in cached_lengths]
                except Exception:
                    continue
                if len(cached_ratios) == len(norm_target):
                    if all(
                        abs(c - t) <= 0.05 for c, t in zip(cached_ratios, norm_target)
                    ):
                        self.data, self.data_split = cached_data
                        self.__class__.datas[cache_key] = ex_split_sig
                        return True, cache_key
        return False, cache_key

    def _read_files_and_split(self, cols, dtm0):
        raw_parts = [[], [], []]
        if isinstance(self.data_path, pd.DataFrame):
            raw_parts[self.set_type].append(self.data_path)
        else:
            use_random_factor = False
            if isinstance(self.data_split, dict):
                data_split_ratios = None
            else:
                data_split_ratios = np.cumsum(
                    [0]
                    + (
                        self.data_split
                        if self.data_split is not None
                        else [0.7, 0.1, 0.2]
                    )
                )
                self.data_split = {}
                use_random_factor = True
            files_list = (
                self.data_path if isinstance(self.data_path, list) else [self.data_path]
            )
            for table in files_list:
                df = pd.read_csv(os.path.join(self.root_path, table)).replace(
                    -9999900, float("nan")
                )
                df = df[~df[dtm0].isna()]
                if not cols["xvsp"] and "spot" in df.columns:
                    df.drop(columns="spot", inplace=True)
                date_clean = df["date"].str.replace("#", "", regex=False)
                df["day_dt"] = pd.to_datetime(date_clean)
                df["day"] = df["day_dt"].dt.date
                if self.query is not None or not cols["xvsp"]:
                    df["day"] = pd.to_datetime(df["date"].str.replace("#", "")).dt.date
                    df["horizon"] = df[f"{dtm0}_{self.in_len}"] - df[dtm0]
                    lasttime = df.groupby(["day"])["date"].max().values
                    df = df[(df["date"].isin(lasttime)) & (df["horizon"] > 0)]
                ds = None
                if self.query is not None:
                    is_oos_mode = "date>" in self.query or isinstance(
                        self.data_path, pd.DataFrame
                    )
                    df = df.query(self.query)
                    if is_oos_mode:
                        ds = [0, 0, 0, len(df) - 1]
                if ds is None:
                    if table in self.data_split:
                        ds = self.data_split[table]
                    else:
                        current_len = len(df)
                        if use_random_factor:
                            ds = (
                                data_split_ratios * uniform(0.8, 0.9) * current_len
                            ).astype(int)
                            print_color(94, f"split{table} at {ds}")
                        else:
                            ds = (data_split_ratios * current_len).astype(int)
                        self.data_split[table] = ds
                try:
                    split_date = (
                        df["date"].iloc[ds]
                        if len(ds) > 0 and isinstance(ds, (list, np.ndarray))
                        else "N/A"
                    )
                    print(f"{table} date at split: {split_date}")
                except Exception as e:
                    print(f"Error printing date at split for {table}: {e}")

                delta = df["day_dt"] - epoch0
                df["date"] = delta.dt.days + delta.dt.seconds / 86400.0

                for i in range(3):
                    start_idx = ds[i]
                    end_idx = ds[i + 1] if i + 1 < len(ds) else None

                    if end_idx is not None:
                        raw_parts[i].append(df.iloc[start_idx:end_idx])
                    else:
                        if i == 0 and len(ds) == 2:
                            raw_parts[0].append(df.iloc[start_idx : ds[1]])
                            raw_parts[2].append(df.iloc[ds[1] :])
                            break

                        if i == len(ds) - 1:
                            raw_parts[i].append(df.iloc[start_idx:])

                if len(ds) == 4 and len(raw_parts[2]) == 0:
                    raw_parts[2].append(df.iloc[ds[3] :])

                if "day_dt" in df.columns:
                    del df["day_dt"]

        df_raws = [
            pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            for parts in raw_parts
        ]

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
            assert len(xnp[-1]) < 1 or (
                np.isnan(xnp[-1]).sum() == 0 and np.isnan(xpc[-1]).sum() == 0
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

        ivs = np.array([vnp.index(v) for v in vvs], dtype=int)
        idat = vnp.index("date")
        idatv = vvs.index("date")

        icyc_indices = np.array([vnp.index(v) for v, c in cols["cyc"]], dtype=int)
        icyc_periods = np.array([c for v, c in cols["cyc"]], dtype=float)

        for i, df_raw in enumerate(df_raws):
            if len(xnp[i]) < 1:
                cyclics.append([])
                xvs.append([])
            else:
                cyclic_other_features_xnp = []
                for idx, period in zip(icyc_indices, icyc_periods):
                    col_data = xnp[i][:, idx]
                    cyclic_other_features_xnp.extend(cyclic_encode(col_data, period))

                c = np.concatenate(
                    cyclic_t(xnp[i][:, idat]) + cyclic_other_features_xnp,
                    axis=1,
                )
                xx, xsp[i], xpc[i] = [
                    scalers[j].transform(x[i]).reshape((-1, self.in_len, x[i].shape[1]))
                    for j, x in enumerate(xs[:3])
                ]
                cyclics.append(
                    np.concatenate(
                        [
                            xx[:, :, ivs],
                            c.reshape((-1, self.in_len, c.shape[1])),
                        ],
                        axis=2,
                    )
                )
                xnp[i] = np.delete(xx, ivs, axis=2)

                x = df_raw[vvs].values
                cyclic_other_features_x = []
                for idx, period in zip(icyc_indices, icyc_periods):
                    col_data = x[:, idx]
                    cyclic_other_features_x.extend(cyclic_encode(col_data, period))

                c = np.concatenate(
                    cyclic_t(x[:, idatv]) + cyclic_other_features_x,
                    axis=1,
                )

                x = (x[:, ivs] - scalers[0].mean_[ivs]) / scalers[0].scale_[ivs]
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
            df_raws = [pd.DataFrame() for _ in range(3)]
            df_raws[self.set_type] = self.data_path
            cache_key = None
        else:
            cache_hit, cache_key = self._check_cache_and_load()
            if cache_hit:
                return
            df_raws = self._read_files_and_split(cols, cols["dtm0"])
        names = data_names(cols, self.in_len)
        arrays = self._extract_initial_arrays(df_raws, cols, names)
        xnp, xsp, xpc, xvsp, xvs, y, cyclics = arrays
        xs = [xnp, xsp, xpc, xvsp]
        scalers = self._prepare_scalers(xs, y, cols)
        arrays = self._apply_transforms(df_raws, scalers, arrays, cols, names)
        xnp, xsp, xpc, xvsp, xvs, y, cyclics = arrays
        assert len(y) >= 3
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
                len(names[6]) // cols["sect"],
            ),
            list(zip([x.shape for x in xnp], xnp, xsp, cyclics, xpc, xvs, xvsp, y)),
        ]
        if cache_key is not None:
            file_sig = cache_key[0]
            split_sig = cache_key[2]
            if split_sig[0] == "dict":
                self.__class__.datas[cache_key] = self.data, self.data_split
            elif split_sig[0] == "list":
                dict_items = []
                for k, v in self.data_split.items():
                    val = tuple(v.tolist()) if isinstance(v, np.ndarray) else v
                    dict_items.append((k, val))
                dict_sig = ("dict", tuple(sorted(dict_items)))
                dict_cache_key = (file_sig, self.data_name, dict_sig, str(self.query))
                self.__class__.datas[dict_cache_key] = self.data, self.data_split
                self.__class__.datas[cache_key] = dict_sig

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
