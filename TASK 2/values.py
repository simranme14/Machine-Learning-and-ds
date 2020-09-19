# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 03:10:03 2020

@author: Simran Agrawal
"""

import requests

url = 'http://127.0.0.1:5000/Model'
r = requests.post(url,json={'hours':9.25})

print(r.json())
