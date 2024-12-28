"""dataset MTS."""

import warnings
import os
import calendar
from random import uniform
from datetime import datetime, timedelta
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

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


# def cyclic(x, c):
#     cyclics = []
#     for v, p in c:
#         cyclics += cyclic_encode(x[:, v], p)
#     return cyclics


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

    def __read_data__(self):
        if self.data_name + str(self.data_path) in self.datas:
            self.data = self.__class__.datas[self.data_name + str(self.data_path)]
            return
        cols = data_columns(self.data_name)
        dtm0 = cols["dtm0"]
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
                    -99999, float("nan")
                )
                df = df[~df[dtm0].isna()]
                if not cols["xvsp"] and "spot" in df.columns:
                    df.drop(columns="spot")
                if self.query is not None or not cols["xvsp"]:
                    df["day"] = pd.to_datetime(df["date"].str.replace("#", "")).dt.date
                    df["horizon"] = df[f"{dtm0}_{self.in_len}"] - df[dtm0]
                    lasttime = df.groupby(["day"])["date"].max().values
                    df = df[(df["date"].isin(lasttime)) & (df["horizon"] > 0)]

                if self.query is not None:
                    df = df.query(self.query)
                    ds = [0, 0, 0, len(df) - 1]
                elif table in self.data_split:
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
        vy, vnp, vnpt, vpc, vvs, vpct, vsp, vspt = data_names(cols, self.in_len)
        xnp, xpc, xvsp, xvs, y, cyclics, xsp = [[] for _ in range(7)]
        for i, df_raw in enumerate(df_raws):
            if len(df_raw.columns) < 1:
                df_raw = pd.DataFrame(columns=df_raws[self.set_type].columns)
                df_raws[i] = df_raw
            xnp.append(df_raw[vnpt].values.astype(float).reshape((-1, len(vnp))))
            xsp.append(df_raw[vspt].values.astype(float).reshape((-1, len(vsp))))
            xpc.append(df_raw[vpct].values.reshape((-1, 1)))
            xvsp.append(
                df_raw["spot"].values.reshape(-1, 1)
                if cols["xvsp"]
                else np.zeros((len(df_raw), 1), float)
            )
            y.append(df_raw[vy].values)

            assert (
                len(xnp[-1]) < 1
                or np.isnan(xnp[-1]).sum() == 0
                and np.isnan(xpc[-1]).sum() == 0
            )
        if self.scaler is None:
            scaler_np, scaler_p, scaler_y, scaler_sp, scaler_vsp = [
                StandardScaler() for _ in range(5)
            ]
            scaler_np.fit(xnp[0])
            scaler_sp.fit(xsp[0].reshape((-1, 1)))
            scaler_p.fit(xpc[0])
            scaler_y.fit(y[0])
            scaler_vsp.fit(xvsp[0])
        else:
            scaler_np, scaler_p, scaler_y, scaler_sp, scaler_vsp = self.scaler
        ivs = [vnp.index(v) for v in vvs]
        idat = [vnp.index(v) for v in ["date"]]
        idatv = [vvs.index(v) for v in ["date"]]
        icyc = [[vnp.index(v) for v, c in cols["cyc"]], [c for v, c in cols["cyc"]]]
        for i, df_raw in enumerate(df_raws):
            if len(xnp[i]) < 1:
                cyclics.append([])
                xvs.append([])
                # xvsp.append([])
            else:
                c = np.concatenate(
                    cyclic_t(xnp[i][:, idat])
                    + cyclic_encode(xnp[i][:, icyc[0]], icyc[1]),
                    axis=1,
                )
                xx = scaler_np.transform(xnp[i]).reshape((-1, self.in_len, len(vnp)))
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
                xsp[i] = scaler_sp.transform(xsp[i].reshape((-1, 1))).reshape(
                    (-1, self.in_len, len(vsp))
                )
                xpc[i] = scaler_p.transform(xpc[i]).reshape((-1, self.in_len, len(vpc)))
                x = df_raw[vvs].values
                c = np.concatenate(
                    cyclic_t(x[:, idatv]) + cyclic_encode(x[:, icyc[0]], icyc[1]),
                    axis=1,
                )
                x = (x - scaler_np.mean_[ivs]) / scaler_np.scale_[ivs]
                xvs.append(np.concatenate([x, c], axis=1))
                xvsp[i] = scaler_vsp.transform(xvsp[i])
                y[i] = (
                    scaler_y.transform(y[i]).reshape(-1, 1, y[i].shape[1]),
                    y[i][:, 0] - 1,
                )
                assert (
                    np.isnan(xnp[i]).sum() == 0
                    and np.isnan(xsp[i]).sum() == 0
                    and np.isnan(xpc[i]).sum() == 0
                    and np.isnan(cyclics[-1]).sum() == 0
                    and np.isnan(xvs[-1]).sum() == 0
                    and np.isnan(xvsp[-1]).sum() == 0
                )
        self.data = [
            (scaler_np, scaler_p, scaler_y, scaler_sp, scaler_vsp),
            (
                xnp[self.set_type].shape[2]
                # + xsp[self.set_type].shape[2]
                + cyclics[self.set_type].shape[2] + xpc[self.set_type].shape[2],
                xvs[self.set_type].shape[1] + xvsp[self.set_type].shape[1],
                cols["ycat"] + len(vy),
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
        self.__class__.datas[self.data_name + str(self.data_path)] = self.data

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
        seq_y = y[0][index], y[1][index]

        return seq_x, seq_y

    def __len__(self):
        return self.shape[0]

    def inverse_transform(self, data):
        dt = data[0].cpu() if isinstance(data, (tuple, list)) else data.cpu()
        w = self.scaler[2].mean_.shape[0]
        dt = np.concatenate(
            (self.scaler[2].inverse_transform(dt[:, 0, :w]), dt[:, 0, w:]), axis=1
        )
        return (dt, data[1].cpu()) if isinstance(data, (tuple, list)) else dt
