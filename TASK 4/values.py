# -*- coding: utf-8 -*-
"""

@author: Simran Agrawal
"""

import requests

urlpath = 'http://127.0.0.1:5000/tree'
values = requests.post(urlpath,json={'SepalLengthCm':5.6, 'SepalWidthCm':2.43, 
                            'PetalLengthCm':4.123, 'PetalWidthCm':1.23})

print(values.json())