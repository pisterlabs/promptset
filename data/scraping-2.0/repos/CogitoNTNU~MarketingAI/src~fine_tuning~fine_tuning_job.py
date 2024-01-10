import openai
from src.config import Config

PATH_TO_DATA = "src/fine_tuning/"

def upload_training_file(file: str) -> str:
    """Uploads a training file to OpenAI and returns the file id"""
    openai.api_key = Config().API_KEY
    response = openai.File.create(file=open(PATH_TO_DATA + file, "rb"), purpose="fine-tune")
    return response

def create_fine_tuning_job(file_id: str, model: str) -> None:
    """Creates a fine tuning job on OpenAI with the given file id and model"""
    openai.api_key = Config().API_KEY
    openai.FineTuningJob.create(training_file=file_id, model=model)
