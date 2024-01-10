import os
import openai
import dotenv  # Import the dotenv module
from time import sleep

# Load environment variables from the .env file
dotenv.load_dotenv()

# Set the API key using os.environ
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"
openai.api_key = os.environ["OPENAI_API_KEY"]

print(f'Fine-tunning ...')
file_id = "FILE_ID"
res = openai.FineTuningJob.create(
    training_file=file_id,
    model="gpt-3.5-turbo"
)
print(f"Training Response: {res}")

job_id = res["id"]
print(f"Job ID: {res}")
while True:
    res = openai.FineTuningJob.retrieve(job_id)
    if res["finished_at"] != None:
        print(f"Doing: {res}")
        break
    else:
        print(".", end="")
        sleep(100)

ft_model = res["fine_tuned_model"]
print(f"Fine Tunning Response: {ft_model}")