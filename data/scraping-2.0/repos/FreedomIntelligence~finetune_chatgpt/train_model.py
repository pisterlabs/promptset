import openai
import time
import logging


openai.api_key = "Your API-KEY"


def configure_logging():
    """
    Configures logging settings.
    """
    logging.basicConfig(filename='output.log', level=logging.INFO,
                        format='%(asctime)s [%(levelname)s]: %(message)s')
    return logging.getLogger()


def upload_file(file_name):
    """
    Uploads a file to OpenAI for fine-tuning.

    :param file_name: Path to the file to be uploaded.
    :return: Uploaded file object.
    """
    # Note: For a 400KB train_file, it takes about 1 minute to upload.
    file_upload = openai.File.create(file=open(file_name, "rb"), purpose="fine-tune")
    logger.info(f"Uploaded file with id: {file_upload.id}")

    while True:
        logger.info("Waiting for file to process...")
        file_handle = openai.File.retrieve(id=file_upload.id)

        if len(file_handle) and file_handle.status == "processed":
            logger.info("File processed")
            break
        time.sleep(60)

    return file_upload


if __name__ == '__main__':
    # Configure logger
    logger = configure_logging()

    file_name = "train_data.jsonl"
    uploaded_file = upload_file(file_name)

    logger.info(uploaded_file)
    job = openai.FineTuningJob.create(training_file=uploaded_file.id, model="gpt-3.5-turbo")
    logger.info(f"Job created with id: {job.id}")

    # Note: If you forget the job id, you can use the following code to list all the models fine-tuned.
    # result = openai.FineTuningJob.list(limit=10)
    # print(result)


