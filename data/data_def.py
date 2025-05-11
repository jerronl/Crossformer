sections = 4
dtms = [f"dtm_{d}" for d in range(sections)]
curve = ["c", "p", "c-10", "c-5", "c5", "c10", "p-10", "p-5", "p5", "p10"]
dcs = {
    "vols": {
        "dtm0": dtms[0],
        "vs": ["date", "e2d", dtms[0], "vol"],
        "crv": curve,
        "vml": [f"{v}_{d}" for d in range(sections) for v in [curve[0]]],
        "vmsp": [f"{v}_{d}" for d in range(sections) for v in curve[1:]],
        "dtm": dtms[1:],
        "cat": [],
        "vpc": ["spot", "close", "hi", "lo"],
        "vnp": ["vs", "dtm", "vml"],
        "vsp": ["vmsp"],
        "cyc": [["e2d", 70], [dtms[0], 5]],
        "opc": ["close", "hi", "lo"],
        "y": ["vml", "vmsp"],
        "ycat": 0,
        "sect": sections,
        "xvsp": True,
    },
    "prcs": {
        "cat": ["pmcat"],
        "y": ["cat", "opc"],
        "ycat": 20,
        "xvsp": False,
    },
}
dic_procs = dcs["vols"].copy()
dic_procs.update(dcs["prcs"])
dcs["prcs"] = dic_procs

def set_cat(cat):
    dcs["prcs"]["ycat"]=cat
    

def data_columns(data):
    return dcs[data]


def extract_cols(cols, names):
    res = []
    for c in names:
        if c in cols:
            res += extract_cols(cols, cols[c])
        else:
            res.append(c)
    return res


def data_names(cols, in_len):
    vnp = extract_cols(cols, ["vnp"])
    vsp = extract_cols(cols, ["vsp"])
    vnpt = [f"{var}_{i}" for i in range(1, in_len + 1) for var in vnp]
    vspt = [f"{var}_{i}" for i in range(1, in_len + 1) for var in vsp]
    vpc = cols["vpc"]
    vvs = cols["vs"]
    vy = extract_cols(cols, ["y"])
    vpct = [f"{var}_{i}" for i in range(1, in_len + 1) for var in vpc]
    return vy, vnp, vnpt, vpc, vvs, vpct, vsp, vspt
