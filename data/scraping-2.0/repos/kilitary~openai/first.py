# 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#  Copyright (c) 2024/Axis9 (Umbrella corp. experimental division grouping style) | kilitry@gmail.com
#  https://linktr.ee/kilitary                                                                      0
#                                                                                                  0
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

import json
import os
import sys

import openai
from pprint import pprint
import requests
import time

openai.api_key = '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'

tokens_all = "2 JP 1-02    carrier frequencies 50,005,016 Hz and 50,005,018 Hz;2 x 700-800MHz 1.0W 4G LTE USA i Phone (AT&T & Verizon)  (Two Bands);2 x 758-830MHz 1.0W 5G, 4G LTE Low (Two Bands);2 x 850-895MHz 1.0W  CDMA 850  (Two Bands);2 x 920-965MHz 1.0W  GSM900  (Two Bands);2 x 1800-1920MHz 1.0W DCS  (Two Bands);2 x 1800-2000MHz 1.0W  DCS, PCS  (Two Bands);2 x 2100-2170MHz 1.0W 3G, UMTS  (Two Bands);2 x 2500-2700MHz 1.0W 4G WiMAX Sprint  (Two Bands);2 x 2570-2690MHz 1.0W 4G LTE High  (Two Bands);3G (WCDMA) 2100 – 2170 Мгц;4G LTE: 2320-2690 Мгц;5.1-5.9GHz 1.0W WiFi 11.a;5G 3400-3600MHz 1.0W 5G LTE;5G 3600-3800MHz 1.0W 5G LTE;96.6Mhz USB-LSB Hopping (98 Milton Street Site);164-183MHz 1.0W  Lojack 164MHz;173MHz 1.0W;315MHz 1.0W Remote Control;400-480MHz 1.0W UHF Remote Control;433MHz 1.0W UHF Remote Control;868MHz: 1.0W Remote Control;1170-1280MHz 1.0W GPS L2 + L5+ Glonass L2;1380Mhz.-1620Mhz.;1450-1620MHz 1.0W 5G+ GPS L1 +Glonass L1;1570-1620 MHz 1.0W GPS L1 + Glonass L1;1700-1800MHz 4G LTE;2300-2500MHz 1.0W 4G LTE + WiFi 11.b & g;2400-2500MHz 1.0W  WiFi 11.bg;CDMA:   870 – 880 Мгц;DCS:    1805 – 1850 Мгц;Exposure levels to selected regions of the brain typically involve peak sound pressures above 100 kPa (194 dB) at 250–500 kHz;GSM:    930 – 960 Мгц;Lumping 170-180 together is a strange idea given the way spectrum in the U.S. is allocated. From 150-ish to 162-ish, we have the normal police, fire, marine, and railroad channels. From 162-174 we have frequencies assigned to different civilian agencies of the federal government. From 174-180 is TV Channel 7.;PCS:    1930 – 1990 Мгц;PHS:    1900 – 1925 Мгц;WiFi:   2400 – 2500 Мгц"

tokens = tokens_all.split(';')
content = ""
models = openai.Model.list()

for token in tokens:
    payload = {
        "model"   : "gpt-3.5-turbo",
        "messages": [
            {"role"   : "system",
             "content": f"info about {token}"
             }
        ]
    }
    
    headers = {
        'Authorization': f"Bearer {openai.api_key}"
    }
    url = "https://api.openai.com/v1/chat/completions"
    print(f'requesting info about {token} ...')
    response = requests.request("POST", url, headers=headers, json=payload).json()
    output = response["choices"][0]["message"]["content"]
    print(output)
    content += f"-> {token}\r\n{output}\r\n\r\n"
    
    with open('full.txt', 'wt') as f:
        f.write(content)
