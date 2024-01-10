import os
import openai
from datasets import Dataset
import logging
import sys
from tqdm.auto import tqdm
import time
openai.organization = "org-JkswNfkhKMfjPPgyLUjElGPH"
openai.api_key = os.getenv("OPENAI_API_KEY")
from models.model_runner import load_test_set

"""

Attention: run this from the root directory of wiki_poc!

This script is used to test the GPT-3.5-turbo model on the datasets.
Mask-Filling is not possible, so we use the ChatCompletion API to prompt for the name referred to as <mask>.
"""

if __name__ == "__main__":

    dataset = sys.argv[1]

    if dataset == "rulings":
        CONFIG = "original"
        ids_file_path = "test_set_ids_rulings.csv"
    elif dataset == "wikipedia":
        CONFIG = "paraphrased"
        ids_file_path = "test_set_ids.csv"

    else:
        raise ValueError("Please provide either 'rulings' or 'wikipedia' as first argument.")

    # ensure required test ids file exists
    assert os.path.exists(ids_file_path), f"{ids_file_path} file not found. Please run generate_test_set_ids.py for your dataset first."
    # ensure api key is set
    assert openai.api_key is not None, "OPENAI_API_KEY environment variable not set."

    MODEL_NAME = "gpt-3.5-turbo-0613"
    PATH = f"{dataset}_predictions_{MODEL_NAME.replace('/', '_')}_{CONFIG}.jsonl"

    user_prompt = """Who is the person refered to as <mask>? Only give the exact name without punctuation.
                     You are not allowed to respond with anything but the name, no more than 3 words. If you don't know,
                     give your best guess. Hint: The name is NOT <mask>."""

    # dataset with initial pages
    dataset = load_test_set(dataset_type=dataset)

    # only process pages which have not been processed yet
    if os.path.exists(PATH):
        # store already processed results in result_dataset
        result_dataset = Dataset.from_json(PATH)
        # get set of page ids which have already been processed
        processed_ids = set(result_dataset['page_id'])
        # filter out pages from dataset which have already been processed
        dataset = dataset.filter(lambda x: x["id"] not in processed_ids)
    
    else:
        result_dataset = Dataset.from_dict({'prediction': [], 'page_id': [], 'input_length': []})

    # iterate over pages in dataset
    for index, page in enumerate(tqdm(dataset)):
        
        # extract text from page
        text = page[f"masked_text_{CONFIG}"][:10000]
        # prompt openai api for prediction
        try:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    { "role": "user", "content": text + " " + user_prompt },
                ],
                temperature=0.5,
                max_tokens=10,
                top_p=1,
                n=1,
                frequency_penalty=0,
                presence_penalty=1,
                stop=["\n"]
            )
        except Exception as e:
            logging.error(e)
            # sleep a few seconds, then try again
            time.sleep(5)
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    { "role": "user", "content": text + " " + user_prompt },
                ],
                temperature=0.5,
                max_tokens=10,
                top_p=1,
                n=1,
                frequency_penalty=0,
                presence_penalty=1,
                stop=["\n"]
            )

        # add prediction to result dataset
        result_dataset = result_dataset.add_item(
            {'prediction': response['choices'][0]['message']['content'],
             'page_id': page['id'], 'input_length': len(text + " " + user_prompt)})
    
        # periodically save file
        if index % 100 == 0:
            logging.info('Saving dataset intermediately to path %s', PATH)
            result_dataset.to_json(PATH)
        
        # sleep for 5 seconds to avoid rate limit
        time.sleep(5)

    # save dataset
    logging.info('Saving dataset to path %s', PATH)
    result_dataset.to_json(PATH)