import openai
from dotenv import load_dotenv
from datetime import datetime, timedelta
from loguru import logger
import os
import warnings
import tqdm
import pandas as pd

from src.utils import preprocess_email, ElasticSearchClient, XMLReader
from src.config import ES_CLOUD_ID, ES_USERNAME, ES_PASSWORD, ES_INDEX

warnings.filterwarnings("ignore")
load_dotenv()

# if set to True, it will use chatgpt model ("gpt-3.5-turbo") for all the completions
CHATGPT = True

# COMPLETION_MODEL - only applicable if CHATGPT is set to False
OPENAI_ORG_KEY = os.getenv("OPENAI_ORG_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.organization = OPENAI_ORG_KEY
openai.api_key = OPENAI_API_KEY


if __name__ == "__main__":

    ### VARIABLES TO BE CONFIGURED BY USER
    # file path to be saved for prepared dataset
    CSV_FILE_PATH = "data.csv"

    if os.path.exists(CSV_FILE_PATH):
        os.remove(CSV_FILE_PATH)

    # if APPLY_DATE_RANGE is set to False, elasticsearch will fetch all the docs in the index
    APPLY_DATE_RANGE = False

    ### DATA COLLECTION
    xml_reader = XMLReader()
    elastic_search = ElasticSearchClient(es_cloud_id=ES_CLOUD_ID, es_username=ES_USERNAME,
                                         es_password=ES_PASSWORD)

    dev_urls = [
        "https://lists.linuxfoundation.org/pipermail/bitcoin-dev/",
        "https://lists.linuxfoundation.org/pipermail/lightning-dev/"
    ]

    for dev_url in dev_urls:
        logger.info(f"dev_url: {dev_url}")

        if APPLY_DATE_RANGE:
            current_date_str = None
            if not current_date_str:
                current_date_str = datetime.now().strftime("%Y-%m-%d")
            start_date = datetime.now() - timedelta(days=7)
            start_date_str = start_date.strftime("%Y-%m-%d")
            logger.info(f"start_date: {start_date_str}")
            logger.info(f"current_date_str: {current_date_str}")

            docs_list = elastic_search.extract_data_from_es(ES_INDEX, dev_url, start_date_str, current_date_str)

        else:
            docs_list = elastic_search.fetch_all_data_for_url(ES_INDEX, dev_url)

        dev_name = dev_url.split("/")[-2]
        logger.success(f"Total threads received for {dev_name}: {len(docs_list)}")

        # docs_list = docs_list[:100]  # for testing on small dataset
        dataset = []

        for doc in tqdm.tqdm(docs_list):
            res = None
            try:
                email_body = doc['_source'].get('body')
                email_summary = doc['_source'].get('summary')

                if email_body and email_summary:
                    preprocessed_email_body = preprocess_email(email_body=email_body)

                    B_INST, E_INST = "[INST]", "[/INST]"
                    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                    assistant_instruct = "You are an intelligent assistant."

                    template = f"""<s>{B_SYS}{assistant_instruct}{E_SYS}</s> {B_INST} ### Input: Suppose you are a programmer and you are enriched by programming knowledge. You will be going through other programmers mail sent to you and you will be extracting all the important information out of the mail and composing a blog post. Even if the mail is divided into parts and parts, your extraction summary should not be in bullet points. It should be in multiple paragraphs. I repeat, never in bullet points. You have to follow some rules while giving a detailed summary. 
                    The rules are below:
                        1. While extracting, avoid using phrases referring to the context. Instead, directly present the information or points covered.  Do not introduce sentences with phrases like: "The context discusses...", "In this context..." or "The context covers..." or "The context questions..." etc
                        2. The summary tone should be formal and full of information.
                        3. Add spaces after using punctuation and follow all the grammatical rules.
                        4. Try to retain all the links provided and use them in proper manner at proper place.
                        5. The farewell part of the email should be completely ignored.
                        6. Ensure that summary is not simply a rephrase of the original content with minor word changes, but a restructured and simplified rendition of the main points.
                        7. Most importantly, this extracted information should be relative of the size of the email. If it is a bigger email, the extracted summary can be longer than a very short email. 
                        Context: {preprocessed_email_body}\n

                        ### Output: {E_INST} {email_summary}"""

                    dataset.append({"text": template})

                else:
                    logger.warning(f"Email body: {bool(email_body)}, Email Summary: {bool(email_summary)}")

            except Exception as ex:
                error_message = f"Error occurred: {ex}"
                if res:
                    error_message += f", Response: {res}"
                logger.error(error_message)

        df = pd.DataFrame(dataset)
        df.to_csv(CSV_FILE_PATH, index=False)
        logger.success(f"CSV file generated successfully: {CSV_FILE_PATH}")
