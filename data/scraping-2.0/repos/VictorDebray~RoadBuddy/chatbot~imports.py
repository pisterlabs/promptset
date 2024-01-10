# Author:  DINDIN Meryll
# Date:    15 September 2019
# Project: RoadBuddy

import os
import json
import torch
import boto3
import tarfile
import torch.nn.functional as F
import numpy as np
import requests

from itertools import chain
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel
from pytorch_pretrained_bert import OpenAIGPTTokenizer