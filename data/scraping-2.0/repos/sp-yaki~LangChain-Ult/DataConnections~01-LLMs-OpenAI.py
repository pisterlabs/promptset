from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

api_key = os.getenv("OPENAI_API_KEY")

# Initialize the LLM
llm = OpenAI(openai_api_key=api_key)

messages1 = "Translate this sentence from English to French. I love programming."
print(messages1)
print(llm(messages1))

messages2 = "Translate this sentence from English to Lao. I love programming."
result = llm.generate([messages1, messages2])

print(result.generations[1][0].text)

