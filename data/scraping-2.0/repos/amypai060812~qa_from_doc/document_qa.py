##########document_qa.py###########
import getpass
import os

import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader
from langchain_llm import *

embedding_function = SentenceTransformerEmbeddings(
	model_name="all-MiniLM-L6-v2")

###

def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)

###

def build_rag_chain_from_doc(
	document_path,
	):


	## loader

	print(f"loading {document_path}")

	loader = TextLoader(document_path)
	docs = loader.load()

	## split

	print(f"spliting the document {document_path}")

	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=1000, 
		chunk_overlap=200)
	splits = text_splitter.split_documents(docs)


	## embedding

	print(f"embedding the document {document_path}")

	vectorstore = Chroma.from_documents(
		documents=splits, 
		embedding=embedding_function,
		)

	retriever = vectorstore.as_retriever()

	## llm

	print(f"loading model")

	llm = CustomLLM(n=128)

	## chain

	print(f"buidling the RAG chain.")

	prompt = hub.pull("rlm/rag-prompt")

	rag_chain = (
		{"context": retriever | format_docs, "question": RunnablePassthrough()}
		| prompt
		| llm
		| StrOutputParser()
	)

	return rag_chain




'''

from document_qa import *

## qa

document_path = "/code/test.txt"

rag_chain = build_rag_chain_from_doc(
	document_path,
	)

rag_chain.invoke("What is large language model?")


# 'A large language model (LLM) is a type of artificial neural network that uses massive amounts of data to learn billions of parameters during training and consumes large computational resources during its training and operation. LLMs are capable of achieving general-purpose language understanding and generation. They are pre-trained using self-supervised learning and semi-supervised learning and work by taking an input text and repeatedly predicting the next token or word. Larger sized models, such as GPT-3, can be prompt-engineered to achieve similar results.'

'''

##########document_qa.py###########