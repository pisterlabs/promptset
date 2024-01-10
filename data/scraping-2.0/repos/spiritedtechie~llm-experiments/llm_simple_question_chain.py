import os

from dotenv import load_dotenv

load_dotenv(".env")

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# model_name = gpt-3.5-turbo, text-davinci-003
# llm = OpenAI(
#     model_name="text-davinci-003",
#     temperature=0,
#     openai_api_key=os.getenv("OPENAI_API_KEY"),
# )

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

#
# Simple LLM chain
#
template = """Question: {question}
Limit the answer to 50 words."""
prompt = PromptTemplate(template=template, input_variables=["question"])
question = "What is an large language model?"
llm_chain = LLMChain(prompt=prompt, llm=llm)
print("text-davinci-003 response -->")
print(llm_chain.run(question))