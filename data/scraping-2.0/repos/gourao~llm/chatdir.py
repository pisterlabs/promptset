import os
import sys
import environ
import pdb

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores import Chroma
from langchain.vectorstores import Milvus

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

env = environ.Env()
environ.Env.read_env()

# Load API key and document location
OPENAI_API_KEY = env('OPENAI_API_KEY')

if OPENAI_API_KEY == "":
	print("Missing OpenAPI key")
	exit()

print("Using OpenAPI with key ["+OPENAI_API_KEY+"]")

path = sys.argv[1]
if path == "":
	print("Missing document path")
	exit()

# Document loading
loader = DirectoryLoader(path, glob="*")
data = loader.load()
print("Loading done")

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(
	chunk_size = 500, 
	chunk_overlap = 0
)

all_splits = text_splitter.split_documents(data)
print("Splitting done")

# Create retriever

vectorstore = Chroma.from_documents(
	documents=all_splits, 
	embedding=OpenAIEmbeddings()
)
'''
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
vectorstore = Milvus.from_documents(
	all_splits,
	embedding=OpenAIEmbeddings(),
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
)
'''

print("Vector store created")

template = """Use the following pieces of context to answer the question at the end. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Connect to LLM for generation
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
qa_chain = RetrievalQA.from_chain_type(
	llm,
	retriever=vectorstore.as_retriever(),
	chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
print("QA chain created")


# prompt loop
def get_prompt():
	print("Type 'exit' to quit")

	while True:
		prompt = input("Enter a prompt: ")

		if prompt.lower() == 'exit':
			print('Exiting...')
			break
		else:
			try:
				# pdb.set_trace()
				result = qa_chain({"query": prompt})
				print(result["result"])
			except Exception as e:
				print(e)

get_prompt()
