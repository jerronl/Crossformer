dtms = [f"dtm_{d}" for d in range(4)]
curve = ["c", "p", "c-10", "c-5", "c5", "c10", "p-10", "p-5", "p5", "p10"]
dcs = {
    "vols": {
        "dtm0": dtms[0],
        "vs": ["date", "e2d", dtms[0], "vol"],
        "v": [f"{v}" for v in curve],
        "vm": [f"{v}_{d}" for d in range(4) for v in curve],
        "dtm": dtms[1:],
        "cat": [],
        "vpc": ["spot", "close", "hi", "lo"],
        "date": ["date"],
        "vnp": ["vs", "dtm", "vm"],
        "cyc": [["e2d", 70], [dtms[0], 5]],
        "opc": ["close", "hi", "lo"],
        "y": ["v"],
        "ycat": 0,
        "xvsp": True,
    },
    "prcs": {
        "cat": ["pmcat"],
        "y": ["cat", "opc"],
        "ycat": 22,
        "xvsp": False,
    },
}
dic_procs = dcs["vols"]
dic_procs.update(dcs["prcs"])
dcs["prcs"] = dic_procs


def data_columns(data):
    return dcs[data]


def data_names(cols, in_len):
    vnp = [c for s in cols["vnp"] for c in cols[s]]  # vnp.sort()
    vnpt = [f"{var}_{i}" for i in range(1, in_len + 1) for var in vnp]
    vpc = cols["vpc"]
    vvs = cols["vs"]
    vy = [c for s in cols["y"] for c in cols[s]]
    vpct = [f"{var}_{i}" for i in range(1, in_len + 1) for var in vpc]
    return vnp, vnpt, vpc, vvs, vy, vpct
