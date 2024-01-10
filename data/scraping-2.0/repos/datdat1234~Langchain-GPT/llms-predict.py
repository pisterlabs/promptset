from langchain.llms import OpenAI
import os
import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

llm = OpenAI(temperature = 0.9)

text = "Where is Ho Chi Minh city?"

print(llm(text))