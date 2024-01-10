import os
import openai
import json

from pdb import set_trace as st

import itertools

from descriptor_strings import stringtolist

import sys
sys.path.append("..") 
from prompts import *

import json
def load_json(filename):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)

old_dict = load_json('descriptors/PACS/descriptors_pacs_ex_domain.json') 

class_list = list(old_dict.keys())
domain_list = list(old_dict['dog'].keys())

feature_list = []
new_dict = old_dict


for cls in class_list:
    for domain in domain_list:
        feature_list.append(old_dict[cls][domain]) #7*11

for i, cls in enumerate(class_list):
    for j, domain in enumerate(domain_list):
        st()
        new_dict[cls][domain] = feature_list[j*7+i]

st()