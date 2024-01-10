# for env
import os
from dotenv import find_dotenv, load_dotenv
# GPT LLM
import openai
# Langchain to feed in private data
# open gpt
from langchain.chat_models import ChatOpenAI
# open file
from langchain.document_loaders import PyPDFLoader
# chat with your data by wrap LLM
from langchain.chains.question_answering import load_qa_chain

# activate the env function
# can use decouple library to import config
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# OpenAI API configs for Langchain
# Per Langchain docs
llm_model = "gpt-4-1106-preview"
# Per Langchain docs
# Temperature config on how creative GPT will output
llm = ChatOpenAI(temperature=0.0, model=llm_model)


# load the data
pf_loader = PyPDFLoader('./docs/frederec_im.pdf')
documents = pf_loader.load()


# create the chain with verbose to expand logging detail
chain = load_qa_chain(llm, verbose=True)
query = 'What year did Fred last use Python?'
# input_documents and question are chain input paramenters
response = chain.run(input_documents=documents,
                     question=query)

print(response)







