#!/usr/bin/env python3
import openai
import numpy as np
import configparser
import requests
import re
import argparse
import pathlib
import os
import json
import operator
from urllib.request import Request, urlopen
from datetime import datetime

config = configparser.ConfigParser()
config.read('.config')
r1 = requests.get('https://api.semanticscholar.org/datasets/v1/release').json()
print(r1[-3:])


r2 = requests.get('https://api.semanticscholar.org/datasets/v1/release/latest').json()
print(r2['release_id'])


print(json.dumps(r2['datasets'][0], indent=2))

r3 = requests.get('https://api.semanticscholar.org/datasets/v1/release/latest/dataset/abstracts').json()
print(json.dumps(r3, indent=2))

    