import logging
import os
import sys
import time

import openai

sys.path.append("../..")
from src.loader import load_data


# logging settings
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="../../../logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    datefmt="%y-%m-%d %H:%M:%S"
)

DATASETSPATH = "../../datasets"

import warnings
warnings.filterwarnings("ignore")


# bash command: export OPEN_AI_KEY=INSERT_KEY_HERE
openai.api_key = os.environ["OPEN_AI_KEY"]


def lemmatize_query(x_test, dname):
    """Query the OpenAI API to predict lemmata of a list of sentences."""
    lemmata = []
    tokens = 0
    with open(f'../../nbs/gpt3_outputs/gpt3-{dname}.txt', 'w',
              encoding='utf-8') as f:
        for sent in x_test:
            try:
                prompt = f"Lemmatisiere die Tokenliste: {sent}"
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=len(prompt)
                )
                answer = response["choices"][0]["text"]
                tokens += response['usage']['total_tokens']
                f.write(answer+'\n')
                time.sleep(3.)  # prevent rate limit errors
            except Exception as err:
                logger.error(err)
                logger.info(f"{tokens} tokens used.")
    print(f"{tokens} tokens used for {dname}.")
    return lemmata


# lemmatize all datasets
for x_test, y_test, z_test, z_test_xpos, dname in load_data(DATASETSPATH):
    lemmatize_query(x_test, dname)
