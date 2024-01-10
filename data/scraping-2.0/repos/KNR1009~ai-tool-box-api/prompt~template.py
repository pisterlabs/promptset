from langchain import PromptTemplate
from langchain.llms import OpenAI

import os
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

template = """
{subject}のレシピを数字の箇条書きでかんたんに教えてください。
"""

prompt = PromptTemplate(
		template=template,
    input_variables=["subject"]
)
prompt_text = prompt.format(subject="カレー")
print(prompt_text)

llm = OpenAI(openai_api_key=openai_api_key, model_name="text-davinci-003")
print(llm(prompt_text))