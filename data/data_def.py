def data_columns(data):
    return {'vols':{
        'vs':[  'date', 'e2d', 'dtm0',      ],
        'vm':[ f'{v}{d}' for d in range(4)
              for v in 
              [ 'level','slope','curve',]   ], 
        'dtm':[f'dtm{d}' for d in range(1,4)],
        'cat':[                             ],
        'vpc':[ 'spot','close', 'hi', 'lo', ],
        'date':['date',                     ],
        'vnp':[ 'vs','dtm','vm',            ],
        'cyc':[['e2d', 70] ,['dtm0', 5],    ],
        'opc':[ 'close', 'hi', 'lo',        ],
        'y' : [ 'vm',                       ],
        'ycat':0,
        'xvsp':True,
    },'prcs':{
        'vs':[  'date', 'e2d', 'dtm0',      ],
        'vm':[ f'{v}{d}' for d in range(4)
              for v in 
              [ 'level','slope','curve',]   ], 
        'dtm':[f'dtm{d}' for d in range(1,4)],
        'cat':[ 'pmcat',                    ],
        'vpc':[ 'spot','close', 'hi', 'lo', ],
        'date':['date',                     ],
        'vnp':[ 'vs','dtm','vm',            ],
        'cyc':[['e2d', 70] ,['dtm0', 5],    ],
        'opc':[ 'close', 'hi', 'lo',        ],
        'y' : [ 'cat', 'opc',               ],
        'ycat':22,
        'xvsp':False,
    },
    }[data]


def data_names(cols, in_len):
    vnp = [c for s in cols["vnp"] for c in cols[s]]  # vnp.sort()
    vnpt = [f"{var}_{i}" for i in range(1, in_len + 1) for var in vnp]
    vpc = cols["vpc"]
    vvs = cols["vs"]
    vy = [c for s in cols["y"] for c in cols[s]]
    vpct = [f"{var}_{i}" for i in range(1, in_len + 1) for var in vpc]
    return vnp, vnpt, vpc, vvs, vy, vpct
