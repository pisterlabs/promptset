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
from log_info import logger

a = 1
chroma = Chroma()

embedding_function = SentenceTransformerEmbeddings(
	model_name="all-mpnet-base-v2", model_kwargs= {"device": "cuda:2"})

prompt = hub.pull("rlm/rag-prompt")

###

def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)

###

# def build_share():
# 	loader_share = TextLoader('./uploaded/share.txt')
# 	docs_share = loader_share.load()
# 	text_splitter = RecursiveCharacterTextSplitter(
# 		chunk_size=1000, 
# 		chunk_overlap=200)
	
# 	splits_share = text_splitter.split_documents(docs_share)
# 	vectorstore_share = chroma.from_documents(
# 			documents=splits_share, 
# 			embedding=embedding_function,
# 			persist_directory= f"./chroma_db/share")

# 	vectorstore_share.persist()
# 	logger.info('Add to shared chromadb Success!')
# 	return vectorstore_share

def build_rag_chain_from_doc(
	document_path,
	text_name,
	text,
	):

	## loader

	print(f"loading {document_path}")

	loader = TextLoader(document_path)
	docs = loader.load()

	
	## split

	print(f"spliting the document {document_path}")

	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=3000, 
		chunk_overlap=200)
	splits = text_splitter.split_documents(docs)
	


	## embedding

	print(f"embedding the document {document_path}")
	# try:
	vectorstore = chroma.from_documents(
		documents=splits, 
		embedding=embedding_function,
		persist_directory= f"./chroma_db/{text_name}")
	# vectorstore = chroma.from_documents(
	# 	documents=splits, 
	# 	embedding=embedding_function,
	# 	persist_directory= f"./chroma_db/{text_name}")
	vectorstore.persist()
	logger.info('Create and save chromadb Success!')
	# build_share()
	# if os.path.exists("./chroma_db/share"):
		
	# 	# try:
	# 		# vectorstore_share = Chroma(persist_directory=f"./chroma_db/share", embedding_function=embedding_function)
	# 		# vectorstore_share.add_texts(texts=text, embedding_function=embedding_function)
	# 	vectorstore_share = Chroma.from_documents(
	# 		documents=splits_share, 
	# 		embedding=embedding_function,
	# 		persist_directory= f"./chroma_db/share")
	# 	vectorstore_share.persist()
	# 	logger.info('Add to shared chromadb Success!')
	# 	# except Exception as e:
	# 	# 	logger.info('Add to shared chromadb FAIL!')
	# else:
	# 	vectorstore_share = Chroma.from_documents(
	# 	documents=splits_share, 
	# 	embedding=embedding_function,
	# 	persist_directory= f"./chroma_db/share"
	# 	)
	# 	vectorstore_share.persist()
	# 	logger.info('Add to shared chromadb Success!2')

	return "Success"
	# except Exception as e:
	# 	logger.info('Create and save chromadb FAIL!')
	# 	return "Fail"

def answer_from_doc(text_name, question):
	# try:
	
	logger.info(f"Text name answer part: {text_name}")
	if text_name == 'share':
		loader_share = TextLoader('./uploaded/share.txt')
		docs_share = loader_share.load()
		text_splitter = RecursiveCharacterTextSplitter(
			chunk_size=1000, 
			chunk_overlap=400)
		
		splits_share = text_splitter.split_documents(docs_share)
		vectorstore = chroma.from_documents(
				documents=splits_share, 
				embedding=embedding_function,
				persist_directory= f"./chroma_db/share")
	else:
		logger.info(f"text_name : {text_name}")
		vectorstore = Chroma(persist_directory=f"./chroma_db/{text_name}", embedding_function=embedding_function)
	# except:
	# 	logger.info(f"Chromadb {text_name} not exist | RECREATing")
	# 	save_folder = 'uploaded'
	# 	save_path = os.path.join(save_folder, text_name+".txt")
	# 	vectorstore = build_rag_chain_from_doc(save_path, text_name)
	retriever = vectorstore.as_retriever()
	
	print(retriever)

	## llm

	print(f"loading model")

	llm = CustomLLM(n=128)

	## chain

	print(f"buidling the RAG chain.")

	rag_chain = (
		{"context": retriever | format_docs, "question": RunnablePassthrough()}
		| prompt
		| llm
		| StrOutputParser()
	)

	response = rag_chain.invoke(question)
	if question in response:
		response = response.replace(question, '')
	if response != "I don't know" or "I don't know" not in response:
		relat_doc = retriever.get_relevant_documents(question)[0].page_content
		# print(relat_doc[0].page_content)
	else:
		relat_doc = ''

	return response, relat_doc




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