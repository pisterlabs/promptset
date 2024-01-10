
import openai
import os

from dotenv import load_dotenv

from pieutils.config import CHATGPT_FINETUNING

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

dataset = 'ae-110k' # oa-mine
directory_path_preprocessed_mave = f'{CHATGPT_FINETUNING}/{dataset}'
data_path = f'{directory_path_preprocessed_mave}/ft_chatgpt_description_with_example_values-{dataset}.jsonl'

response = openai.File.create(
  file=open(data_path, "rb"),
  purpose='fine-tune',
  user_provided_filename=f'ft_chatgpt_description_with_example_values-{dataset}'
)

print(response)


