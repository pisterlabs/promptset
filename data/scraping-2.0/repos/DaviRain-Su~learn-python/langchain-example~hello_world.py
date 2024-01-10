from langchain.llms import OpenAI
import os

# print(os.environ["OPENAI_API_KEY"])

llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])

text = "What would be a good company name for a company that makes colorful socks?"

print("Question: ", text)
print("Answer: ", llm(text))
