import os
from dotenv import load_dotenv
from jsontojsonl import jsondatapath
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

file_response = openai.File.create(
  file=open(jsondatapath, "rb"),
  purpose='fine-tune'
)
print(file_response)
response = openai.FineTuningJob.create(training_file=file_response["id"], model="gpt-3.5-turbo", suffix="MDGPT")
print(response)