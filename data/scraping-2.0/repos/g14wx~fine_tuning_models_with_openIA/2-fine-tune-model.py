from openai import OpenAI
import os
from dotenv import load_dotenv, set_key

load_dotenv()
key = os.getenv('OPENAI_API_KEY')
training_file_env = os.getenv('CURRENT_TRAINING_FILE_ID')
model_to_train = os.getenv('OPENAI_MODEL_ID')

client = OpenAI(api_key=key)

response = client.fine_tuning.jobs.create(
    training_file=training_file_env,
    model=model_to_train
)

job_id = response.id
set_key(dotenv_path=".env", key_to_set="CURRENT_FINE_TUNING_JOB_ID", value_to_set=job_id)
print(f"Job ID: {job_id}")
