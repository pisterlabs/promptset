from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

import config


from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
openai_api_key = config.api_key
llm=OpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

text_splitter = CharacterTextSplitter()
loader = PyPDFLoader("/Users/tannerpapenfuss/thesis.AI/langchain/Chase-Data-Driven.pdf")
#loader = PyPDFLoader("/Users/tannerpapenfuss/thesis.AI/langchain/Ford-Data-Driven.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

chain = load_summarize_chain(llm, chain_type="map_reduce")
#docs = [Document(page_content=t) for t in texts[4:7]]
print(chain.run(texts))
#print(texts[4:7])