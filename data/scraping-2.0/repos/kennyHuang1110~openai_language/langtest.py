from langchain.chat_models import ChatOpenAI
import os
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

llm =OpenAI(model_name="text-davinci-003", max_tokens=100)

response = llm("陽明山在哪裡")  # Call the language model with the input prompt


generated_content = response.strip()

print(generated_content)

