import logging
import threading
import time
import unicodedata

from textdistance import Levenshtein
from tqdm import tqdm

from constants import *
from logging.handlers import RotatingFileHandler
import sys
import os
import openai
from threading import Thread
from dotenv import load_dotenv
import re

responses = []


load_dotenv('api_key.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

def chatgpt_query(query, model = "gpt-4-0314", temperature=0, replace_newline=True, max_tokens=150):
    response = openai.ChatCompletion.create(
            model=model,
            messages=query,
            temperature=temperature,
            # request_timeout=90, # set timeout in the thread instead of the api call - to reduce charges for timed-out threads
            max_tokens=max_tokens
            )
    if replace_newline:
        return response.choices[0].message["content"].replace('\n', ' ')
    else:
        return response.choices[0].message["content"]


def setup_directories():
    if not os.path.exists('dataset/'):
        os.makedirs('dataset/')
    if not os.path.exists('logs/'):
        os.makedirs('logs/')


def get_logger(log_file, depth=logging.DEBUG):
    logging.basicConfig(filename="logs/{}".format(log_file),
                        filemode='a')

    logger = logging.getLogger()
    logger.setLevel(depth)

    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')

    handler = RotatingFileHandler("logs/{}".format(log_file), maxBytes=1024*1024*5, backupCount=1)
    handler.setFormatter(log_formatter)
    handler.setLevel(depth)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def query_thread(prompts, global_index, model, temp):
    global responses
    count = len(prompts)
    i = 0
    responses_thread = []
    pbar = tqdm(total=count)
    while i < count:
        try:
            query = [
                {"role": "user", "content": prompts[i]}
            ]
            response = chatgpt_query(query, model=model, temperature=temp)
            i += 1
            responses_thread.append(response)
            pbar.update(1)
        except Exception as e:
            print(e)
            time.sleep(10)
    pbar.close()
    responses[global_index] = responses_thread
    print("Thread " + str(i) + "completed ")
    print("==============================================================")


def create_and_run_api_request_threads(queries, n_threads, logger, model='gpt-4-0314', temperature=0):
    global responses

    count = len(queries)
    responses = [[] for _ in range(n_threads)]
    partitions = []
    bin_size = count // n_threads

    for i in range(n_threads - 1):
        partitions.append(queries[i * bin_size: (i + 1) * bin_size])

    partitions.append(queries[(n_threads - 1) * bin_size:])

    threads = []
    for i in range(n_threads):
        threads.append(threading.Thread(target=query_thread, args=(partitions[i], i,model, temperature)))

    logger.info("starting API resquests to OPENAI's GPT 4 using " + str(n_threads) + " threads")
    logger.info("Number of threads created: " + str(len(threads)))
    logger.info("Number of partitions created: " + str(len(partitions)))
    logger.info("Size of each partition: " + str(bin_size))

    for i in range(n_threads):
        threads[i].start()
    for i in range(n_threads):
        threads[i].join(timeout=90) # set timeout in the thread instead of the api call - to reduce charges for timed-out threads

    return responses

def name_match(candidate, responses):
    for i in responses:
        if name_similarity(candidate, i):
            return True
    return False


def normalize_string(input_str):
    # Remove punctuation and extra whitespace
    input_str = re.sub(r'[^\w\s]', '', input_str)
    input_str = ' '.join(input_str.split())

    # Remove diacritics (accent marks)
    input_str = ''.join(c for c in unicodedata.normalize('NFKD', input_str) if not unicodedata.combining(c))

    # Convert to lowercase for case-insensitive comparison
    input_str = input_str.lower()

    return input_str


def tokenize_string(s):
    """Split string into tokens."""
    return s.split()


def is_abbreviation(abbr, word):
    """Check if `abbr` is an abbreviation of `word`."""
    return word.startswith(abbr)

def name_similarity_exact(name1, name2):
    """Calculate similarity score between two names."""
    # Normalizing the names
    norm_name1 = normalize_string(name1)
    norm_name2 = normalize_string(name2)

    # Tokenizing the names
    tokens1 = tokenize_string(norm_name1)
    tokens2 = tokenize_string(norm_name2)

    # Initial match based on abbreviations
    for token1 in tokens1:
        for token2 in tokens2:
            if is_abbreviation(token1, token2) or is_abbreviation(token2, token1):
                return 1

    # Using Levenshtein distance as a similarity metric
    levenshtein = Levenshtein()
    distance = levenshtein.distance(norm_name1, norm_name2)
    max_len = max(len(norm_name1), len(norm_name2))
    similarity = (max_len - distance) / max_len

    return similarity

def name_similarity(name1, name2):
    """Calculate similarity score between two names."""
    return name_similarity_exact(name1, name2) > 0.8  # Threshold can be adjusted
