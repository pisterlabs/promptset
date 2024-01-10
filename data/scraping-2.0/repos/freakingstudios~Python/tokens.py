from langchain.llms import OpenAI
import os
os.environ["OPENAI_API_KEY"] = "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"

llm = OpenAI(temperature=0.9)

text = "What is 2+2"
print(llm(text))
