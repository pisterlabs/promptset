import os
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


file = openai.File.create(
  file=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetuning.jsonl'), "rb"),
  purpose='fine-tune'
)

openai.FineTuningJob.create(training_file=file.id, model="gpt-3.5-turbo")
