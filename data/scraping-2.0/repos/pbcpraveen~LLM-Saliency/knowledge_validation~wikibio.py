import itertools
import sys
import os
from pathlib import Path
import pickle
import pandas as pd
import json
import openai
import pickle
import dotenv

path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))
import propmts
from common_utils import *
from constants import *
from utils import *

def get_verification_flag(row):
    response = row[GPT_4_RESPONSE]
    candidate = row[Attribute.NAME.value]
    return name_match(candidate, [response])


load_dotenv('../api_key.env')
openai.api_key = os.getenv("OPENAI_API_KEY")
setup_directories()
logger = get_logger('wikibio.log')

data = pickle.load(open("../data/dataset/wikibio_raw.pickle", "rb"))
logger.info("Checking if LLM has been already queried")
if "wikibio_LLM_recall.csv" not in os.listdir('dataset/'):
    logger.info("Querying LLM for knowledge validation")
    prompts = [person_prompt(i) for i in data]
    names = [i[CONTEXTUALISING_ATTRIBUTES][Attribute.NAME.value] for i in data]
    index = [i for i in range(len(data))]

    df = pd.DataFrame()
    df[INDEX_COLUMN] = index
    df[Attribute.NAME.value] = names
    df[PROMPT_COLUMN] = prompts

    responses = create_and_run_api_request_threads(prompts, 5, logger)
    responses = list(itertools.chain(*responses))
    df[GPT_4_RESPONSE] = responses

    logger.info('Saving responses to ./dataset/wikibio_LLM_recall.csv')
    df.to_csv("dataset/wikibio_LLM_recall.csv")

logger.info("Reading LLm responses from ./dataset/wikibio_LLM_recall.csv")
responses = pd.read_csv("dataset/wikibio_LLM_recall.csv")
flags = responses.apply(get_verification_flag, axis=1).to_list()

for i in range(len(data)):
    data[i][VERIFIED] = flags[i]

logger.info("Writing data with verified flag to dataset/wikibio_verified.pickle")
with open(f"dataset/{VERIFIED_RECORDS[WIKIBIO]}", 'wb') as handle:
    pickle.dump(data, handle)

