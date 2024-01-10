"""
Author: Joshua Ashkinaze
Date: 2023-06-20

Description: This script uses ChatGPT to predict binary political events from Metaculus
"""

import argparse
import openai
import json
import pandas as pd
import multiprocessing
import numpy as np
import re
import os
import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential, before_sleep_log, before_sleep_nothing

def log_before_sleep(retry_state):
    if retry_state.attempt_number > 1:
        sleep_duration = retry_state.next_action.sleep
        logging.info(f"Sleeping for {sleep_duration} seconds before retry attempt {retry_state.attempt_number}.")
    before_sleep_nothing(retry_state)
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        return np.NaN


@retry(wait=wait_random_exponential(multiplier=30, min=1, max=60), stop=stop_after_attempt(30),
       retry_error_callback=lambda x: logging.info(f"Retry error: {x}"),
       before_sleep=log_before_sleep)
def make_prediction(event, event_id, description, res_criteria, temperature=0):
    try:
        # Set up the base prompt without the description or res_criteria
        prompt = f"""INSTRUCTIONS:
    As an artificial intelligence model, you have been trained on a vast dataset up to September 2021. Your task is to provide an educated guess on a possible future event ("{event}"), based on your training data.

    While making your guess, keep in mind:
    1. Overconfidence can lead to inaccurate forecasts. Be aware of this and provide a conservative estimate.
    2. Consider base rates: Remember to think about how often events have occurred in the past as a guide.
    3. Consider the availability heuristic: People often judge the likelihood of events based on how easily instances come to mind. Be cautious of this and ensure that your prediction is not overly influenced by recent, memorable events.
    4. Adopt an "outside view": Instead of focusing solely on the specific details of the event in question, take a step back and consider similar events or situations. Use the base rates of those events to inform your prediction.
    5. Rare and unpredictable events can and do occur. While these "black swan" events may be unlikely, they should not be ignored entirely.
    6. Acknowledge uncertainty: The future is not set in stone and your guess should take into account the inherent unpredictability of future events.

    QUESTION:
    {event}
    """

        # Include description and resolution criteria if they are not np.nan
        if not (isinstance(description, float) and np.isnan(description)):
            prompt += f"\nDESCRIPTION:\n{description}"

        if not (isinstance(res_criteria, float) and np.isnan(res_criteria)):
            prompt += f"\n\nRESOLUTION DETAILS:\n{res_criteria}\n"

        prompt += """
    RETURN:
    A json file with fields { "answer": "YES" or "NO", "reasoning": Your reasoning for your guess, "confidence": Your confidence level on a scale from 0 to 1 }

    CONSTRAINTS:
    Do not add anything to your answer other than "YES" or "NO", your reasoning, and your confidence level.
    """

        messages = openai.ChatCompletion.create(
            temperature=temperature,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        msg = messages['choices'][0]['message']['content']
        try:
            msg_dict = json.loads(msg.strip())
            msg_dict['id'] = event_id

            return msg_dict
        except Exception as e:
            logging.info(f"Error occured after getting msg: {e}. The returned response was {msg}")
            return {'id': event_id, 'answer': np.NaN, 'reasoning': np.NaN}
    except Exception as e:
        logging.info(f"An error occurred: {e}")
        raise e


def process_event(args):
    q, id, description, res_criteria = args
    with open('../secrets/secrets.json', 'r') as f:
        secrets = json.load(f)
    openai.api_key = secrets['openai_key']
    pred = make_prediction(q, id, description, res_criteria)
    logging.info(pred)
    return pred


def main(debug_mode, num_requests):
    LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO, format=LOG_FORMAT,
                        datefmt='%Y-%m-%d %H:%M:%S', filemode='w')

    logging.info(f"Starting script with parameters debug_mode={debug_mode}, num_requests={num_requests}")
    events = pd.read_csv("../data/metaculus_events.csv")
    if debug_mode:
        events = events.sample(5)

    questions = events['title'].tolist()
    ids = events['id'].tolist()

    events['description_clean'] = events['description'].apply(clean_text)
    events['resolution_criteria_clean'] = events['resolution_criteria'].apply(clean_text)
    descriptions = events['description_clean'].tolist()
    res_criteria = events['resolution_criteria_clean'].tolist()

    n_jobs = multiprocessing.cpu_count()

    responses = []
    for _ in range(num_requests):
        with multiprocessing.Pool(processes=n_jobs) as pool:
            responses += pool.map(process_event, zip(questions, ids, descriptions, res_criteria))
    logging.info(f"Total responses: {len(responses)}")
    responses_df = pd.DataFrame(responses)
    responses_df.to_csv("../data/raw_llm_preds.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metaculus event prediction script")
    parser.add_argument('-d', '--debug_mode', action='store_true', help="Enable debug mode to sample only 5 events")
    parser.add_argument('-n', '--num_requests', type=int, default=1, help="Number of requests per event")
    args = parser.parse_args()

    main(args.debug_mode, args.num_requests)