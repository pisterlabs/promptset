import lmql
import json
from pathlib import Path
import argparse
from functools import partial
import os
import openai
import time

# getting json name
parser = argparse.ArgumentParser()
parser.add_argument('second_argument')
# opening json file
file_path = Path.cwd() / parser.parse_args().second_argument
with file_path.open(mode='r', encoding="utf-8") as f:
    data = json.load(f)

model = "davinci"

# creating output base filename
info_list = parser.parse_args().second_argument.split(".")

print(info_list)
json_name = ".".join([ info_list[1].split("/")[1], info_list[0].split("/")[1], model, info_list[1].split("/")[0]])


print(json_name)