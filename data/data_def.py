def data_columns(data):
    return {'vols':{
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
        'y' : [ 'cat', 'opc', 'vm',         ],
    },
    }[data]
