import os
import openai
import logging
import subprocess
import re
import json
import docker
import csv
import time
from solgpt.solgpt import SolGPT


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - \n%(message)s \n', level=logging.INFO)

class SolGPT_withouthint(SolGPT):
  """
  Overwriting the constants to run the `cmd` on a different folder in the docker container.
  Assuming that in the `without_hint` folder there will be all forders containing the four tries got from ChatGPT
  """
    CURR_DIR = os.path.dirname(__file__)
    VUL_TOOLS = {
            "slither": {
                "docker_name": "slither",
                "docker_image": "trailofbits/eth-security-toolbox",
                "docker_shared_folder": "/home/emanuele/Documents/PhD/Works/2023_Compsac_conf/cleaned:/home/ethsec/shared",
                "cmd": lambda sol_file, pragma: [f"solc-select use {pragma}", f"slither shared/without_hint/{sol_file.split('round')[0][:-2]}/{sol_file} --json -"]
            },
    }

    def __init__(self, sol_path:str, vul_tool:str="slither"):
        super().__init__(sol_path, vul_tool)

class SolGPT_withhint(SolGPT):
  """
  Overwriting the constants to run the `cmd` on a different folder in the docker container.
  Assuming that in the `with_hint` folder there will be all forders containing the four tries got from ChatGPT
  """
    CURR_DIR = os.path.dirname(__file__)
    VUL_TOOLS = {
            "slither": {
                "docker_name": "slither",
                "docker_image": "trailofbits/eth-security-toolbox",
                "docker_shared_folder": "/home/emanuele/Documents/PhD/Works/2023_Compsac_conf/cleaned:/home/ethsec/shared",
                "cmd": lambda sol_file, pragma: [f"solc-select use {pragma}", f"slither shared/with_hint/{sol_file.split('round')[0][:-2]}/{sol_file} --json -"]
            },
    }

    def __init__(self, sol_path:str, vul_tool:str="slither"):
        super().__init__(sol_path, vul_tool)

CURR_DIR = os.path.dirname(__file__)
SC_PATH = os.path.join(os.path.dirname(__file__) ,"cleaned/")
files = os.listdir(SC_PATH)
sc_path = sorted([ff for ff in files if ff.endswith(".sol") and os.path.exists(os.path.join(SC_PATH, ff))])

# Rename file according to the `with hint` or `without hint` folder to analyze
with open(os.path.join(CURR_DIR, "sc_data_without_hintcsv"), "w") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["smartcontract", "pragma", "vulns", "high", "medium", "low", "tot"])
    for sc_file in sc_path:
        logging.info(f"Processing {sc_file} in {os.path.join(SC_PATH, sc_file)}")
        try:
            # Use `SolGPT_withouthint` or `SolGPT_withhint` according to the `with hint` or `without hint` folder to analyze
            sc = SolGPT_withouthint(os.path.join(SC_PATH, sc_file))
            vulns = sc._get_vul()
            high = len(list(filter(lambda x: x["impact"] == "High", vulns)))
            medium = len(list(filter(lambda x: x["impact"] == "Medium", vulns)))
            low = len(list(filter(lambda x: x["impact"] == "Low", vulns)))
            tot = high + medium + low
            writer.writerow([sc_file, sc.pragma, vulns, high, medium, low, tot])
        except Exception as e:
            logging.error(f"Error processing {sc_file}: {e}")
            writer.writerow([sc_file, "", e, 0, 0, 0, 0])
