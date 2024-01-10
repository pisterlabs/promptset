import time
import openai
from datetime import datetime, timedelta
from loguru import logger
import os
from dotenv import load_dotenv
import warnings
import tqdm
import json

from src.utils import preprocess_email, ElasticSearchClient, XMLReader
from src.config import ES_CLOUD_ID, ES_USERNAME, ES_PASSWORD, ES_INDEX
from jsonl_data_stats import get_jsonl_data_stats, check_jsonl_data_format

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

    ### VARIABLES TOBE CONFIGURE AS PER USER'S NEED
    # if APPLY_DATE_RANGE is set to False, elasticsearch will fetch all the docs in the index
    APPLY_DATE_RANGE = False
    # path for the file to be saved after data preparation
    JSONL_FILE_PATH = "data.jsonl"
    # number of epochs you want to fune-tune the model
    NUM_EPOCHS = 1
    # custom suffix you want to add to name a new fine-tuning model (not more than 18 characters)
    FINE_TUNING_MODEL_SUFFIX = "custom-suffix"
    # add the fine-tuned model name if you want to add more data to the fine-tuned model
    FINETUNING_MODEL_NAME = "gpt-3.5-turbo"
    # 'fine_tuned_model' from https://platform.openai.com/docs/api-reference/fine-tuning/retrieve

    if os.path.exists(JSONL_FILE_PATH):
        os.remove(JSONL_FILE_PATH)

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

        # docs_list = docs_list[:10]  # for testing on small dataset

        ### DATA PREPARATION (INTO JSONL FORMAT)
        for doc in tqdm.tqdm(docs_list):
            res = None
            try:
                email_body = doc['_source'].get('body')
                email_summary = doc['_source'].get('summary')

                if email_body and email_summary:
                    preprocessed_email_body = preprocess_email(email_body=email_body)

                    prompt_with_context = f"""Suppose you are a programmer and you are enriched by programming knowledge. You will be going through other programmers mail sent to you and you will be extracting all the important information out of the mail and composing a blog post. Even if the mail is divided into parts and parts, your extraction summary should not be in bullet points. It should be in multiple paragraphs. I repeat, never in bullet points. You have to follow some rules while giving a detailed summary. 
                    The rules are below:
                        1. While extracting, avoid using phrases referring to the context. Instead, directly present the information or points covered.  Do not introduce sentences with phrases like: "The context discusses...", "In this context..." or "The context covers..." or "The context questions..." etc
                        2. The summary tone should be formal and full of information.
                        3. Add spaces after using punctuation and follow all the grammatical rules.
                        4. Try to retain all the links provided and use them in proper manner at proper place.
                        5. The farewell part of the email should be completely ignored.
                        6. Ensure that summary is not simply a rephrase of the original content with minor word changes, but a restructured and simplified rendition of the main points.
                        7. Most importantly, this extracted information should be relative of the size of the email. If it is a bigger email, the extracted summary can be longer than a very short email. 
                    \n\nCONTEXT:\n\n{preprocessed_email_body}"""

                    messages = [
                        {"role": "system", "content": "You are an intelligent assistant."},
                        {"role": "user", "content": f"{prompt_with_context}"},
                        {"role": "assistant", "content": f"{email_summary}"},
                    ]

                    with open(JSONL_FILE_PATH, "a") as outfile:
                        messages_data = {"messages": messages}
                        json.dump(messages_data, outfile)
                        outfile.write('\n')

                else:
                    logger.warning(f"Email body: {bool(email_body)}, Email Summary: {bool(email_summary)}")

            except Exception as ex:
                error_message = f"Error occurred: {ex}"
                if res:
                    error_message += f", Response: {res}"
                logger.error(error_message)

        logger.success(f"JSONL file generated successfully!: {JSONL_FILE_PATH}")

        ### CHECK JSONL DATA FORMATTING
        logger.info(f"checking jsonl data formatting...")
        data_report = check_jsonl_data_format(file_path=JSONL_FILE_PATH)

        ### JSONL PRICE STATS
        get_jsonl_data_stats(file_path=JSONL_FILE_PATH, n_epochs=NUM_EPOCHS)

        ### FINE-TUNING PROCESS
        # no errors in dataformat, move on to fine-tuning steps
        if data_report:
            logger.info(f"uploading the file for fine-tuning...")

            # UPLOAD THE FILE
            upload_response = openai.File.create(
                file=open(JSONL_FILE_PATH, "rb"),
                purpose='fine-tune'
            )
            logger.info(f"upload_response: {upload_response}")
            '''
            here is the sample upload response:
            {
              "id": "file-abc123",
              "object": "file",
              "bytes": 140,
              "created_at": 1613779121,
              "filename": "mydata.jsonl",
              "purpose": "fine-tune",
              "status": "uploaded" | "processed" | "pending" | "error"
            }
            '''

            # get uploaded file id
            uploaded_file_id = upload_response['id']
            logger.success(f"upload successful with file_id : {uploaded_file_id}")

            logger.info(f"please wait 15 seconds, while the uploaded file gets ready...")
            time.sleep(15)

            # FOR PREVIOUSLY FINE-TUNED MODEL
            if FINETUNING_MODEL_NAME.startswith("ft"):
                logger.info(f"initiating the fine-tuning job on existing model: {FINETUNING_MODEL_NAME}")
                create_finetune_response = openai.FineTune.create(
                    training_file=uploaded_file_id,
                    model=FINETUNING_MODEL_NAME,
                    # n_epochs=NUM_EPOCHS,
                    suffix=FINE_TUNING_MODEL_SUFFIX
                )

            # FOR NEW FINE-TUNING
            else:
                logger.info(f"initiating the fine-tuning job on new model: {FINETUNING_MODEL_NAME}")
                fine_tuning_create_response = openai.FineTuningJob.create(
                    training_file=uploaded_file_id,
                    model=FINETUNING_MODEL_NAME,
                    # n_epochs=NUM_EPOCHS,
                    suffix=FINE_TUNING_MODEL_SUFFIX
                )
                '''
                here is the sample response of creating fine-tuning job:
                {
                  "object": "fine_tuning.job",
                  "id": "ft-AF1WoRqd3aJAHsqc9NY7iL8F",
                  "model": "gpt-3.5-turbo-0613",
                  "created_at": 1614807352,
                  "fine_tuned_model": null,
                  "organization_id": "org-123",
                  "result_files": [],
                  "status": "pending",
                  "validation_file": null,
                  "training_file": "file-abc123",
                }
                '''

                finetune_id = fine_tuning_create_response['id']
                logger.success(f"fine-tune job id: {finetune_id}")
                logger.info("please wait 60 seconds, before retrieving the fine-tune status...")

                time.sleep(60)
                logger.info(f"Fine-tuning info: \n{openai.FineTune.retrieve(id=finetune_id)}")

                logger.info(
                    "Fine-tuning might take a while, run list_finetune_events.py to check the detailed fine-tuning events."
                    "Please note down the 'id' to future use.")

        else:
            logger.warning(f"JSONL data format not as per the standard guidelines. "
                           f"Please refer https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset")
