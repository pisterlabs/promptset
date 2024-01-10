import os

from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

FAISS_MODEL_PATH = 'models/trained'
QUERY = 'What is the article name?'

embeddings = OpenAIEmbeddings()
db = FAISS.load_local(FAISS_MODEL_PATH, embeddings)
docs = db.similarity_search(QUERY)

chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
answear = chain.run(input_documents=docs, question=QUERY)

if " I don't know." == answear:
  print('no answear')

print(answear)

