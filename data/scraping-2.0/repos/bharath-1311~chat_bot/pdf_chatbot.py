from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import pickle
import json
import os

import pypdf
loader = PyPDFLoader("gao-21-26sp.pdf")
data = loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs=text_splitter.split_documents(data)

from langchain.embeddings.openai import OpenAIEmbeddings
with open('config.json') as f:
	config = json.load(f)['embedding']
embeddings = OpenAIEmbeddings(openai_api_key=config['openai'])

from langchain.vectorstores import Pinecone
import pinecone
with open('config.json') as f:
	config = json.load(f)['vectordb']
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', config["pinecone_key"])
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', config["pinecone_env"])
pinecone.init(
api_key=PINECONE_API_KEY,
environment=PINECONE_API_ENV
)
index_name = config['pinecone_index']
docsearch=Pinecone.from_texts([t.page_content for t in docs], embeddings, index_name=index_name)

db = Pinecone.from_existing_index(index_name, embeddings)


with open('Llama-13B-GGUF.pkl', 'rb') as f:
	embeddings = pickle.load(f)
docs=text_splitter.split_documents(data)

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
from langchain.chains import ConversationalRetrievalChain
qa_chain = ConversationalRetrievalChain.from_llm(
llm,
db.as_retriever(search_kwargs={'k': 2}),
return_source_documents=True)
import sys
chat_history = []
while True:
	query = input('Prompt: ')
	if query.lower() in ["exit", "quit", "q"]:
		print('Exiting')
		sys.exit()
	result = qa_chain({'question': query, 'chat_history': chat_history})
	print('Answer: ' + result['answer'])
	chat_history.append((query, result['answer']))
