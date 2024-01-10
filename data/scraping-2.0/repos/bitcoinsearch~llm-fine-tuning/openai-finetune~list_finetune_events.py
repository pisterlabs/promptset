import openai
from pprint import pprint
import os
from dotenv import load_dotenv
import warnings
from loguru import logger

warnings.filterwarnings("ignore")
load_dotenv()

OPENAI_ORG_KEY = os.getenv("OPENAI_ORG_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.organization = OPENAI_ORG_KEY
openai.api_key = OPENAI_API_KEY

# ENTER THE FINE-TUNING JOB ID (i.e., "ftjob-8fVPvtgokefrgSFRdvyGIAzc")
FINETUNING_JOB_ID = "ftjob-8fVPvtgokefrgSFRdvyGIAzc"


def cancel_finetune(fine_tuning_job_id):
    cancel_ft = openai.FineTuningJob.cancel(fine_tuning_job_id)
    logger.info(cancel_ft)


def delete_uploaded_file(file_id):
    # file_id = "file-U0gadd9sL2JAShVh8oEaKZXJ"
    pprint(openai.File.delete(file_id))
    pprint(openai.File.list())


if __name__ == "__main__":

    try:
        # retrieve fine-tune status
        retrieved_ft = openai.FineTuningJob.retrieve(FINETUNING_JOB_ID)

        if retrieved_ft["status"] == "succeeded":
            logger.info(retrieved_ft)
            logger.success("Fine-Tuning Successful!")
            logger.success(f"Fine-tuned model name: {retrieved_ft['fine_tuned_model']}")

        else:
            pprint(openai.FineTuningJob.list_events(FINETUNING_JOB_ID))
            logger.info(f"Fine-Tuning Status: {retrieved_ft['status']}")

    except Exception as ex:
        logger.error(ex)

    '''
    # use below code to cancel the fine-tuning job
    cancel_finetune(FINETUNING_JOB_ID)
    '''
