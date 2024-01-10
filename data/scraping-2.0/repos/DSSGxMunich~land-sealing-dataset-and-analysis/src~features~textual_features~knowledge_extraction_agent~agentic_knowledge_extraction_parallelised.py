import json
import os
from multiprocessing import Process

import openai
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from agentic_knowledge_extractor import extract_knowledge_from_df

# set paths
BASE_DIR = os.path.join("src", "features", "textual_features", "knowledge_extraction_agent")
INPUT_FOLDER_PATH = os.path.join("data", "nrw", "features", "fuzzy_search", "parallel")
OUTPUT_FOLDER_PATH = os.path.join("data", "nrw", "features", "knowledge_extraction_agent")

# set relevant variables
ID_COLUMN = 'filename'
TEXT_COLUMN = 'content'

# set nested dictionary
with open(os.path.join(BASE_DIR, 'keyword_dict_agent.json')) as f:
    AGENT_KEYWORDS = json.load(f)

# set api key from .env file
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


# set up function per process
def process_agent_batch(id_column_name,
                        text_column_name,
                        keyword_dict):
    """ Function to process a batch of agent searches in parallel. It saves the results to a json file

    This function is used to parallelize the agent search.

    Args:
        id_column_name (str): name of the column with the id
        text_column_name (str): name of the column with the text
        keyword_dict (dict): dictionary with the keyword, keyword_short and template_name

    """

    # read in file
    filename = keyword_dict['filename']
    input_df = pd.read_csv(f'{INPUT_FOLDER_PATH}/{filename}', names=[ID_COLUMN, TEXT_COLUMN])

    # apply function for fuzzy search
    result_df = extract_knowledge_from_df(input_df=input_df,
                                          id_column_name=id_column_name,
                                          text_column_name=text_column_name,
                                          keyword_dict=keyword_dict)

    # save as json
    result_json = result_df.to_json(orient='records')
    with open(os.path.join(OUTPUT_FOLDER_PATH, keyword_dict['keyword_short'] + ".json"), "w") as outputfile:
        outputfile.write(result_json)

    logger.info(f"Done with {keyword_dict['keyword_short']}")


# set up function to run processes in parallel
def run_agent():
    """ Function to run the agent in parallel. It saves the results to multiple json file

    Note: This function is not used in the pipeline, as it is not necessary to run the agent in parallel.
        It starts multiple processes, each of which runs the agent on a different keyword. So it is only
        useful if you want to run the agent on multiple keywords at the same time. Also, it does not consider
        the number of cores available on the machine, so it might be more efficient to run the agent on a single
        keyword at a time.
    """
    processes = []

    for keyword_dict in AGENT_KEYWORDS:
        process = Process(target=process_agent_batch, args=(ID_COLUMN, TEXT_COLUMN, keyword_dict))
        processes.append(process)
        process.start()

    # wait for all processes to complete
    for process in processes:
        process.join()


if __name__ == '__main__':
    logger.info(f"Let's go!")
    run_agent()
    logger.info(f"Agent done!")
