from openai import OpenAI
import os

from utils.gptReqest import gptRequest
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv("../.env"))

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# print(client.files.create(
#     file=open("./dataset.jsonl", "rb"),
#     purpose="fine-tune"
# ))


# client.fine_tuning.jobs.create(
#     training_file="file-amSSguysjnwTTMcc86HW35DL",
#     model="gpt-3.5-turbo-1106"
# )


print(client.fine_tuning.jobs.list(limit=10))
