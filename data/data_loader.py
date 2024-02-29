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


def cyclic_encode(original, period):
    return [
        np.sin(2 * np.pi * original / period),
        np.cos(2 * np.pi * original / period),
    ]


def cyclic_t(x):
    if x.dtype == np.float:
        epoch0 = datetime(1899, 12, 31)
        t = np.vectorize(lambda x: (epoch0 + timedelta(days=x)).timetuple())(x)
    else:
        t=np.vectorize(lambda x: 
            datetime(int(x[1:5]),int(x[6:8]),int(x[9:11]),int(x[12:14]),
                     int(x[15:17]),int(x[18:20]),).timetuple())(x)
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
        # self.var_len = size[1]
        # init
        # self.scale = scale
        # self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path
        self.data_split = np.cumsum([0] + data_split)
        # self.scale_statistic = scale_statistic
        self.data_name = data_name
        self.__read_data__()

    def __read_data__(self):
        if self.data_name in self.datas:
            return
        df_raws = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
        for table in self.data_path:
            df = pd.read_csv(os.path.join(self.root_path, table))
            df["date"] = np.vectorize(
                lambda x: datetime(int(x[1:5]),int(x[6:8]),int(x[9:11]),int(x[12:14]),
                                   int(x[15:17]),int(x[18:20]),))(df["date"])
            df=df.replace(-99999,float('nan'))
            for i, df_raw in enumerate(df_raws):
                df_raws[i] = pd.concat(
                    [df_raw,
                     df.iloc[int(self.data_split[i] * len(df)):
                         int(self.data_split[i + 1] * len(df))],])
        # df_raws = [df_raw.sort_values(by="date") for df_raw in df_raws]
        cols = data_columns(self.data_name)
        vnp = list(set([c for s in cols["vnp"] for c in cols[s]]))
        vnp.sort()
        vnpt = [f"{var}_{i}" for i in range(1, self.in_len + 1) for var in vnp]
        xnp = [df_raw[vnpt].values.reshape((-1,len(vnp)))
               for df_raw in df_raws]
        idat = [vnp.index(v) for v in cols["date"]]
        icyc = [[vnp.index(v), c] for v, c in cols["cyc"]]
        cyclics = []
        for x in xnp:
            cyclics.append(cyclic_t(x[:,idat]) + cyclic(x,icyc))

        scaler_np = StandardScaler()
        scaler_np.fit(xnp[0])
        xnp = [scaler_np.transform(xx).reshape((-1,self.in_len,len(vnp)))
               for xx in xnp]
        vpc = cols["vpc"]
        vpct= [f"{var}_{i}" for i in range(1, self.in_len + 1) for var in vpc]
        xpc = [df_raw[vpct].values.reshape((-1)) for df_raw in df_raws]
        scaler_p = StandardScaler()
        scaler_p.fit(xpc[0])
        xpc = [scaler_p.transform(xx).reshape((-1,self.in_len,len(vpc)))
               for xx in xpc]
        ivs = [vnp.index(v) for v in cols["vs"]]
        xvs = [
            (df_raw[cols["vs"]].values - scaler_np.mean[ivs]) / scaler_np.std[ivs]
            for df_raw in df_raws
        ]
        xvsp= [scaler_p.transform(df_raw['spot'].values) for df_raw in df_raws]

        y= [df_raw[cols["vm"]+cols['cat']].values for df_raw in df_raws]
        scaler_y=StandardScaler()
        scaler_y.fit(y[0])
        y = [scaler_y.transform(xx) for xx in y]
        self.__class__.datas[self.data_name] = [
            (scaler_np,scaler_p,scaler_y,),
            (idat,icyc,ivs,),
            list(zip(
                [df_raw.shape for df_raw in df_raws],
                xnp,
                xpc,
                xvs,
                xvsp,
                y,
            )),
        ]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.in_len
        r_begin = s_end
        r_end = r_begin + self.out_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.in_len - self.out_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
