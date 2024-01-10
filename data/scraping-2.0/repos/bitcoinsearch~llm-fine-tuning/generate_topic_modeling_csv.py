import re
import openai
from datetime import datetime, timedelta
from loguru import logger
import os
from dotenv import load_dotenv
import warnings
import tqdm
import pandas as pd
import traceback
import ast
import sys
import time
from openai.error import APIError, PermissionError, AuthenticationError, InvalidAPIType, ServiceUnavailableError

from src.config import ES_CLOUD_ID, ES_USERNAME, ES_PASSWORD, ES_INDEX
from src.utils import preprocess_email, ElasticSearchClient, tiktoken_len, clean_text, split_prompt_into_chunks, empty_dir

warnings.filterwarnings("ignore")
load_dotenv()

OPENAI_ORG_KEY = os.getenv("OPENAI_ORG_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.organization = OPENAI_ORG_KEY
openai.api_key = OPENAI_API_KEY

# logs automatically rotate log file
os.makedirs("logs", exist_ok=True)
logger.add(f"logs/generate_topics_modeling.log", rotation="23:59")


def generate_topics_for_text(text, topic_list):
    logger.info(f"generating keywords ... ")
    topic_modeling_prompt = f"""Analyze the following content and extract the relevant keywords from the provided TOPIC_LIST.
    The keywords should only be selected from the given TOPIC_LIST and match the content of the text.
    TOPIC_LIST = {topic_list} \n\nCONTENT: {text}
    \nBased on these guidelines:
    1. Only keywords from the TOPIC_LIST should be used.
    2. Output should be a Python list of relevant keywords from the TOPIC_LIST that describe the CONTENT.
    3. If the provided CONTENT does not contain any relevant keywords from the given TOPIC_LIST output an empty Python List ie., [].
    \nPlease provide the list of relevant topics:"""

    time.sleep(2)
    logger.info(f"token length: {tiktoken_len(topic_modeling_prompt)}")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with classifying content into specific "
                                          "topics. Your function is to extract relevant keywords from a given text, "
                                          "based on a predefined list of topics. Remember, the keywords you identify "
                                          "should only be ones that appear in the provided topic list."},
            {"role": "user", "content": f"{topic_modeling_prompt}"},
        ],
        temperature=0.0,
        max_tokens=300,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=1.0
    )
    response_str = response['choices'][0]['message']['content'].replace("\n", "").strip()
    response_str = response_str.replace("The relevant topics for the given content are:", "").strip()
    logger.info(f"generated Keywords for this chunk: {response_str}")
    return response_str


def get_keywords_for_text(text_chunks, topic_list):
    logger.info(f"Number of chunks: {len(text_chunks)}")
    keywords_list = []
    for prompt in text_chunks:
        try:
            keywords = generate_topics_for_text(prompt, topic_list)

            if keywords == "[]" or keywords == "['']":
                continue

            # keywords = re.sub(r"(\w)'s", r'\1\'s', keywords)
            # keywords = re.sub(r"(\w)'t", r'\1\'t', keywords)
            keywords = re.sub(r"(\w)'(\w)", r'\1\'\2', keywords)

            if keywords.startswith("['") and not (keywords.endswith("']") or keywords.endswith('"]')):
                logger.warning(f"Model hallucination: {keywords}")

                if keywords.endswith("',"):
                    keywords = keywords[:-1] + "]"

                elif keywords.endswith("', '"):
                    keywords = keywords[:-3] + "]"

                elif keywords.endswith("'"):
                    keywords = keywords + "]"

                elif keywords.endswith("',..."):
                    keywords = keywords[:-4] + "]"

                elif keywords.endswith("', ...]"):
                    keywords = keywords[:-6] + "]"

                else:
                    keywords = keywords + "']"

                logger.warning(f"Keywords after fix: {keywords}")

            elif not keywords.startswith("['") and not keywords.endswith("']"):
                logger.warning(f"Elif: {keywords}")
                continue

            if isinstance(keywords, str):
                keywords = ast.literal_eval(keywords)
            else:
                logger.warning(f"Keywords Type: {keywords}")

            keywords_list.extend(keywords)

        except openai.error.RateLimitError as rate_limit:
            logger.error(f'Rate limit error occurred: {rate_limit}')
            sys.exit(f"{rate_limit}")

        except openai.error.InvalidRequestError as invalid_req:
            logger.error(f'Invalid request error occurred: {invalid_req}')

        except (APIError, PermissionError, AuthenticationError, InvalidAPIType, ServiceUnavailableError) as ex:
            logger.error(f'Other error occurred: {str(ex)}')

    logger.success(f"Generated keywords: {keywords_list}")
    return list(set(keywords_list))


def get_primary_and_secondary_keywords(keywords_list, topic_list):
    clean_keywords_list = [clean_text(i) for i in keywords_list]
    clean_topic_list = [clean_text(i) for i in topic_list]
    primary_keywords = [i for i, j in zip(keywords_list, clean_keywords_list) if j in clean_topic_list]
    secondary_keywords = [i for i, j in zip(keywords_list, clean_keywords_list) if j not in clean_topic_list]
    logger.success(f"Primary Keywords: {len(primary_keywords)}, Secondary Keywords: {len(secondary_keywords)}")
    return primary_keywords, secondary_keywords


def apply_topic_modeling(text, topic_list):
    text_chunks = split_prompt_into_chunks(text)
    keywords_list = get_keywords_for_text(text_chunks=text_chunks, topic_list=topic_list)
    primary_keywords, secondary_keywords = get_primary_and_secondary_keywords(keywords_list=keywords_list,
                                                                              topic_list=topic_list)
    return primary_keywords, secondary_keywords


if __name__ == "__main__":

    delay = 3
    btc_topics_list = pd.read_csv("btc_topics.csv")
    btc_topics_list = btc_topics_list['Topics'].to_list()

    elastic_search = ElasticSearchClient(es_cloud_id=ES_CLOUD_ID, es_username=ES_USERNAME,
                                         es_password=ES_PASSWORD)

    dev_urls = [
        "https://lists.linuxfoundation.org/pipermail/lightning-dev/",
        "https://lists.linuxfoundation.org/pipermail/bitcoin-dev/",
        # "all_data",  # uncomment this line if you want to generate topic modeling on all docs
    ]

    for dev_url in dev_urls:

        if dev_url == "all_data":
            dev_name = "all_data"
            dev_url = None
        else:
            dev_name = dev_url.split("/")[-2]

        logger.info(f"dev_url: {dev_url}")
        logger.info(f"dev_name: {dev_name}")

        # if APPLY_DATE_RANGE is set to False, elasticsearch will fetch all the docs in the index
        APPLY_DATE_RANGE = False

        # if UPDATE_ES_SIMULTANEOUSLY set to True, it will update topics in the elasticsearch docs as we generate them
        UPDATE_ES_SIMULTANEOUSLY = False

        # if SAVE_CSV is set to True, it will store generated topics data into csv file
        SAVE_CSV = True
        SAVE_AT_MULTIPLE_OF = 50

        OUTPUT_DIR = "gpt_output"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        CSV_FILE_PATH = f"{OUTPUT_DIR}/topic_modeling_{dev_name}.csv"

        if APPLY_DATE_RANGE:
            current_date_str = None
            if not current_date_str:
                current_date_str = datetime.now().strftime("%Y-%m-%d")
            start_date = datetime.now() - timedelta(days=7)
            start_date_str = start_date.strftime("%Y-%m-%d")
            logger.info(f"start_date: {start_date_str}")
            logger.info(f"current_date_str: {current_date_str}")
        else:
            start_date_str = None
            current_date_str = None

        elastic_search = ElasticSearchClient(es_cloud_id=ES_CLOUD_ID, es_username=ES_USERNAME,
                                             es_password=ES_PASSWORD)

        docs_list = elastic_search.fetch_data_for_empty_keywords(ES_INDEX, dev_url, start_date_str, current_date_str)
        logger.success(f"TOTAL THREADS RECEIVED WITH AN EMPTY KEYWORDS: {len(docs_list)}")

        if docs_list:

            if os.path.exists(CSV_FILE_PATH):
                stored_df = pd.read_csv(CSV_FILE_PATH)
                logger.info(f"Shape of stored df: {stored_df.shape}")

                stored_source_ids = stored_df['source_id'].to_list()
                logger.info(f"Docs in stored df: {len(stored_source_ids)}")
            else:
                logger.info(f"CSV file path does not exist! Creating new one: {CSV_FILE_PATH}")
                stored_df = pd.DataFrame(columns=['primary_topics', 'secondary_topics', 'source_id'])
                stored_source_ids = stored_df['source_id'].to_list()

            for idx, doc in enumerate(tqdm.tqdm(docs_list)):
                doc_source_id = doc['_source']['id']

                if CSV_FILE_PATH:
                    if doc_source_id in stored_source_ids:
                        continue

                doc_id = doc['_id']
                doc_index = doc['_index']
                logger.info(f"Doc Id: {doc_id}, Source Id: {doc_source_id}")

                doc_body = doc['_source'].get('summary')
                if not doc_body:
                    doc_body = doc['_source'].get('body')
                    doc_body = preprocess_email(email_body=doc_body)

                if not doc['_source'].get('primary_topics'):
                    doc_text = ""
                    if doc_body:
                        doc_title = doc['_source'].get('title')
                        doc_text = doc_title + "\n" + doc_body

                    if doc_text:
                        primary_kw, secondary_kw = [], []
                        try:
                            primary_kw, secondary_kw = apply_topic_modeling(text=doc_text, topic_list=btc_topics_list)

                            if SAVE_CSV and not UPDATE_ES_SIMULTANEOUSLY:
                                row_data = {
                                    'primary_topics': primary_kw if primary_kw else [],
                                    'secondary_topics': secondary_kw if secondary_kw else [],
                                    'source_id': doc_source_id if doc_source_id else None
                                }
                                row_data = pd.Series(row_data).to_frame().T
                                stored_df = pd.concat([stored_df, row_data], ignore_index=True)

                                if idx % SAVE_AT_MULTIPLE_OF == 0:
                                    stored_df.drop_duplicates(subset='source_id', keep='first', inplace=True)
                                    stored_df.to_csv(CSV_FILE_PATH, index=False)
                                    time.sleep(delay)
                                    logger.info(f"csv file saved at IDX: {idx}, PATH: {CSV_FILE_PATH}")

                            elif UPDATE_ES_SIMULTANEOUSLY and not SAVE_CSV:
                                # update primary keyword
                                elastic_search.es_client.update(
                                    index=doc_index,
                                    id=doc_id,
                                    body={
                                        'doc': {
                                            "primary_topics": primary_kw if primary_kw else []
                                        }
                                    }
                                )
                                # update secondary keyword
                                elastic_search.es_client.update(
                                    index=doc_index,
                                    id=doc_id,
                                    body={
                                        'doc': {
                                            "secondary_topics": secondary_kw if secondary_kw else []
                                        }
                                    }
                                )

                            elif SAVE_CSV and UPDATE_ES_SIMULTANEOUSLY:
                                # update primary keyword
                                elastic_search.es_client.update(
                                    index=doc_index,
                                    id=doc_id,
                                    body={
                                        'doc': {
                                            "primary_topics": primary_kw if primary_kw else []
                                        }
                                    }
                                )
                                # update secondary keyword
                                elastic_search.es_client.update(
                                    index=doc_index,
                                    id=doc_id,
                                    body={
                                        'doc': {
                                            "secondary_topics": secondary_kw if secondary_kw else []
                                        }
                                    }
                                )

                                # store in csv file
                                row_data = {
                                    'primary_topics': primary_kw if primary_kw else [],
                                    'secondary_topics': secondary_kw if secondary_kw else [],
                                    'source_id': doc_source_id if doc_source_id else None
                                }
                                row_data = pd.Series(row_data).to_frame().T
                                stored_df = pd.concat([stored_df, row_data], ignore_index=True)

                                if idx % SAVE_AT_MULTIPLE_OF == 0:
                                    stored_df.drop_duplicates(subset='source_id', keep='first', inplace=True)
                                    stored_df.to_csv(CSV_FILE_PATH, index=False)
                                    time.sleep(delay)
                                    logger.info(f"csv file saved at IDX: {idx}, PATH: {CSV_FILE_PATH}")

                            else:  # not SAVE_CSV and not UPDATE_ES_SIMULTANEOUSLY
                                pass

                        except Exception as ex:
                            logger.error(f"Error: apply_topic_modeling: {traceback.format_exc()}")

                            stored_df.drop_duplicates(subset='source_id', keep='first', inplace=True)
                            stored_df.to_csv(CSV_FILE_PATH, index=False)
                            time.sleep(delay)
                            logger.info(f"csv file saved at IDX: {idx}, PATH: {CSV_FILE_PATH}")

                    else:
                        logger.warning(f"Body Text not found! Doc ID: {doc_id}")

            stored_df.drop_duplicates(subset='source_id', keep='first', inplace=True)
            stored_df.to_csv(CSV_FILE_PATH, index=False)
            time.sleep(delay)
            logger.success(f"FINAL CSV FILE SAVED AT PATH: {CSV_FILE_PATH}")

        logger.info(f"Process completed for dev_url: {dev_url}")
