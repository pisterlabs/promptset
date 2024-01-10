import logging
import os
from solgpt.solgpt import SolGPT
import openai
import time


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - \n%(message)s \n', level=logging.INFO)

CURR_DIR = os.path.dirname(__file__)
SC_PATH = os.path.join(CURR__DIR, "cleaned")
files = os.listdir(SC_PATH)
sc_files = sorted([ff for ff in files if ff.endswith(".sol") and os.path.exists(os.path.join(SC_PATH, ff))])[::-1]


# Number of new Solidity code we want to get from ChatGPT
ntimes=4

# Number of times we try to get an answer from ChatGPT if the server is overloaded
max_try = 5

for count, sc_file in enumerate(sc_files):
    sc = SolGPT(sol_path=os.path.join(SC_PATH, sc_file))
    logging.info(f"Proessing {sc_file} ({count+1}/{len(sc_files)})")
    error_count = 0
    for cc in range(1, ntimes+1):
        new_sc_name = f"{sc_file.split('.')[0]}_{cc}round.sol"
        logging.info(f"try {cc}/{ntimes}: {new_sc_name} ")
        new_code = sc.get_fix(with_tool=True, level="Medium", new_sol_file=new_sc_name)
        while error_count < max_try and (isinstance(new_code, openai.error.APIConnectionError) or isinstance(new_code, openai.error.RateLimitError)):
            new_code = sc.get_fix(with_tool=True, level="Medium",new_sol_file=new_sc_name)
            logging.info(f"Failed to write {new_sc_name} the {error_count}\nError {type(new_code)}")
            error_count += 1
            time.sleep(1)
