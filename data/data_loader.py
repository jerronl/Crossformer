"""dataset MTS."""

import warnings
import os
import pandas as pd, numpy as np
from utils.tools import StandardScaler
from einops import rearrange
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from data.data_def import data_columns
import calendar

warnings.filterwarnings("ignore")


epoch0 = datetime(1899, 12, 31)
def excel_date(x):
    delta = datetime(int(x[1:5]),int(x[6:8]),int(x[9:11]),int(x[12:14]),
                     int(x[15:17]),int(x[18:20]),) - epoch0
    return float(delta.days) + (float(delta.seconds) / 86400)

def cyclic_encode(original, period):
    return [
        np.sin(2 * np.pi * original / period),
        np.cos(2 * np.pi * original / period),
    ]

def cyclic_t(x):
    assert np.issubdtype(x.dtype, float)
    
    t = np.vectorize(lambda x: (epoch0 + timedelta(days=x)).timetuple())(x)
    tm_yday = cyclic_encode(
        t[7] - 1, np.vectorize(lambda x: 365 - 28 + calendar.monthrange(x, 2)[1])(t[0])
    )
    tm_mday = cyclic_encode(
        t[2] - 1, np.vectorize(lambda x, y: calendar.monthrange(x, y)[1])(t[0], t[1])
    )
    tm_wday = cyclic_encode(t[6], 7)
    return tm_yday + tm_mday + tm_wday

def cyclic(x,c):
    cyclics=[]
    for v,p in c:
        cyclics+=cyclic_encode(x[:,v],p)
    return cyclics

class DatasetMTS(Dataset):
    datas = {}

    def __init__(
        self,
        root_path,
        data_path,
        data_name,
        in_len,
        flag="train",
        # size=None,
        data_split=None,
        # scale=True,
        # scale_statistic=None,
    ):
        if data_split is None:
            data_split = [0.7, 0.1, 0.2]
        # if size is None:
        #     size = [20, 22]
        type_map = {"train": 0, "val": 1, "test": 2}
        assert flag in type_map
        self.set_type = type_map[flag]
        # info
        self.in_len = in_len
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = np.cumsum([0] + data_split)
        # self.scale_statistic = scale_statistic
        self.data_name = data_name
        self.__read_data__()
        self.shape=self.data[2][self.set_type][0]
        self.scaler=self.data[0]

    def __read_data__(self):
        if self.data_name in self.datas:
            self.data = self.__class__.datas[self.data_name]
            return
        df_raws = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
        for table in self.data_path:
            df = pd.read_csv(os.path.join(self.root_path, table))
            df["date"] = np.vectorize(excel_date)(df["date"])
            df=df.replace(-99999,float('nan'))
            for i, df_raw in enumerate(df_raws):
                df_raws[i] = pd.concat(
                    [df_raw,
                     df.iloc[int(self.data_split[i] * len(df)):
                         int(self.data_split[i + 1] * len(df))],])
        # df_raws = [df_raw.sort_values(by="date") for df_raw in df_raws]
        cols = data_columns(self.data_name)
        vnp = [c for s in cols["vnp"] for c in cols[s]]        # vnp.sort()
        vnpt = [f"{var}_{i}" for i in range(1, self.in_len + 1) for var in vnp]
        vpc = cols["vpc"]
        vvs = cols["vs"]
        vy = [c for s in cols["y"] for c in cols[s]] 
        vpct= [f"{var}_{i}" for i in range(1, self.in_len + 1) for var in vpc]
        xnp, xpc, xvsp, xvs, y, cyclics= [], [], [], [], [], []
        for df_raw in df_raws:
            xnp.append(df_raw[vnpt].values.reshape((-1,len(vnp))))
            xpc.append(df_raw[vpct].values.reshape((-1)))
            y.append(df_raw[vy].values)
        scaler_np, scaler_p, scaler_y = StandardScaler(), StandardScaler(), StandardScaler(), 
        scaler_np.fit(xnp[0])
        scaler_p.fit(xpc[0])
        scaler_y.fit(y[0])
        ivs = [vnp.index(v) for v in vvs]
        idat = [vnp.index(v) for v in cols["date"]]
        idatv= [vvs.index(v) for v in cols["date"]]
        icyc = [[vnp.index(v) for v,c in cols["cyc"]],
                [ c for v, c in cols["cyc"]]]
        for i, df_raw in enumerate(df_raws):
            c=np.concatenate(cyclic_t(xnp[i][:,idat]) +
                             cyclic_encode(xnp[i][:,icyc[0]],icyc[1]),axis=1)
            xx = scaler_np.transform(xnp[i]).reshape((-1, self.in_len, len(vnp)))
            cyclics.append(np.concatenate([
                xx[:,:,:len(ivs)],
                c.reshape((-1,self.in_len,c.shape[1])),
                ],axis=2))
            xnp[i] = xx[:, :, len(ivs) :]
            xpc[i] = scaler_p.transform(xpc[i]).reshape(
                (-1, self.in_len, len(vpc)))
            x = df_raw[vvs].values
            c = np.concatenate(
                cyclic_t(x[:, idatv]) + cyclic_encode(x[:, icyc[0]], icyc[1]), axis=1
            )
            x = (x - scaler_np.mean[ivs]) / scaler_np.std[ivs]
            xvs.append(np.concatenate([x, c,], axis=1))
            xvsp.append(scaler_p.transform(df_raw["spot"].values).reshape(-1, 1))
            y[i] = (scaler_y.transform(y[i]).reshape(-1,1,y[i].shape[1]),y[i][:,0]-1)
        self.data = [
            (scaler_np, scaler_p, scaler_y, ),
            (idat, icyc, ivs, ),
            list(zip(
                    [x.shape for x in xnp],
                    xnp,
                    cyclics,
                    xpc,
                    xvs,
                    xvsp,
                    y,
                )),
        ]
        self.__class__.datas[self.data_name] = self.data

    def __getitem__(self, index): 
        _, xnp, cyclic, xpc, xvs, xvsp, y, = self.data[2][self.set_type]

        seq_x = xnp[index], cyclic[index], xpc[index], xvs[index], xvsp[index]
        seq_y = y[0][index],y[1][index]

        return seq_x, seq_y

    def __len__(self):
        return self.shape[0]

    def inverse_transform(self, data):
        return self.scaler[2].inverse_transform(data[0]),data[1]
