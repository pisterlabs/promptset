import openai
from .setup import OPEN_AI_KEY
import json
from loguru import logger


def upload_training_data(file_path: str, save_path: str):
    logger.info("Uploading data..." + file_path)

    try:
        result = openai.File.create(
            file=open(file_path, "rb"),
            purpose='fine-tune',
            api_key=OPEN_AI_KEY
        )

        print(result)

        # write result to json file
        json.dump(result, open(save_path, "w"), indent=4)

        return result
    except Exception as err:
        logger.error(f"Error uploading data: {err}")
        exit()