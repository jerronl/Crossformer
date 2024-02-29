def data_columns(data):
    return {'vols':{
        'vs':[  'date', 'e2d',              ],
        'vm':[ f'{v}{d}' for d in range(4)
              for v in 
              [ 'level','slope','curve',]   ], 
        'dtm':[f'dtm{d}' for d in range(4)  ],
        'cat':[ 'pmcat',                    ],
        'vpc':[ 'spot','close', 'hi', 'lo', ],
        'date':['date',                     ],
        'vnp':[ 'vs','vm','dtm',            ],
        'cyc':[['e2d', 70] ,['dtm0', 5],    ],
    },
    }[data]
