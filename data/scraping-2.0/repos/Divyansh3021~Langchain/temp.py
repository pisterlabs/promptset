import os

os.environ["OPENAI_API_KEY"] = "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"

from langchain.llms import openai

llm = openai(temperature = 0.9)

# llm = openai()
# chat_model = ChatOpenAI()

llm("What are the top 5 fastest cars in the world ?")
