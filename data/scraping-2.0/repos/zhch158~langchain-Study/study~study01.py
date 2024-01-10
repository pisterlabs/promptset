import os
from utils import setup_workdir
from utils import env
from langchain.llms import OpenAI
import openai

question = "What would be a good company name for a company that makes colorful socks?"
# context = ""
# model="text-davinci-003"
# max_tokens=150
# response = openai.Completion.create(
#     prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
#     temperature=0,
#     max_tokens=max_tokens,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0,
#     stop=None,
#     model=model,
# )
# print(response)

llm = OpenAI(temperature=0)
print(llm(question))
