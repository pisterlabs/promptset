from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)

embedding_function = OpenAIEmbeddings()
db_connection = Chroma(persist_directory='../US_Constitution/',embedding_function=embedding_function)

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm,chain_type='stuff')
question = "What is the 15th amendment?"

docs = db_connection.similarity_search(question)
print(chain.run(input_documents=docs,question=question))

chain = load_qa_with_sources_chain(llm,chain_type='stuff')
print(chain.run(input_documents=docs,question=question))