from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
import openai
import sys
sys.path.append('../..')

#from dotenv import load_dotenv, find_dotenv
#_ = load_dotenv(find_dotenv()) # read local .env file

api_key = ""
api_base = "https://cgaddam-openai.openai.azure.com/"
#deployment = 'cgaddamllm'
openai.api_version = '2023-05-15' # may change in the future
openai.api_type = 'azure'

import os

current_directory = os.getcwd()
print("Current Directory:", current_directory)

loader = CSVLoader(file_path='pokemon.csv')
#pages = loader.load_and_split(page_limit=10)
pages = loader.load()
len(pages)

page = pages[0]
print(page.page_content[0:500])
page.metadata

from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(
    openai_api_key=api_key,
    openai_api_base=api_base,
    openai_api_version='2023-05-15',
    engine='cgaddamembeddings'
)

from langchain.vectorstores import Chroma
persist_directory = 'docs/chroma'
#!rm -rf ./docs/chroma

for i in range(0, len(pages), 10):
  vectordb = Chroma.from_documents(
      documents = pages[i:i+10],
      embedding = embedding,
      persist_directory = persist_directory
  )
  if i%100 == 0:
    print(vectordb._collection.count())
  i = i + 10
  
vectordb.persist()

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=api_key,
    openai_api_base=api_base,
    engine='cgaddamllm'
)


from langchain.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = "Do you know grass type pokemons?"
result = qa({"question": question})
result['answer']

question = "what are the next generations of these pokemons"
result = qa({"question": question})
result['answer']

question = "what are the stats of one of the pokemons you mentioned?"
result = qa({"question": question})
print(result['answer'])