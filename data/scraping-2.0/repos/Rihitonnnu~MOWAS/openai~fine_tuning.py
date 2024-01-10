import os
import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]
# file = openai.File.create(
#     file=open("mydata.jsonl", "rb"),
#     purpose='fine-tune'
# )
openai.FineTuningJob.create(
    training_file="file-GQ717qTCTvvGgfxCJtbJafKy", model="gpt-3.5-turbo")
