import os
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

import config

os.environ["HTTP_PROXY"] = config.HTTP_PROXY
os.environ["HTTPS_PROXY"] = config.HTTPS_PROXY
openai_api_key = config.OPENAI_API_KEY

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=2, openai_api_key=openai_api_key)
with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print(result)
    print(cb)
