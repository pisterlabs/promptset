from openai import OpenAI
import os
from dotenv import load_dotenv, set_key

load_dotenv()
key = os.getenv('OPENAI_API_KEY')
train_file_asset_filename = os.getenv('TRAIN_FILE_ASSET_FILENAME')
client = OpenAI(api_key=key)
response = client.files.create(
    file=open(f"assets/{train_file_asset_filename}", "rb"),
    purpose="fine-tune"
)

file_id = response.id
set_key(dotenv_path=".env", key_to_set="CURRENT_TRAINING_FILE_ID", value_to_set=file_id)
print(f"File uploaded successfully with ID: {file_id}")
