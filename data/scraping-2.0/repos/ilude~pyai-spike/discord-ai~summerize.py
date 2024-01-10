#! /usr/bin/python
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import json

from langchain.chains.summarize import load_summarize_chain

json_file = 'general.json'

with open(json_file) as json_data:
  data = json.load(json_data)

docs = []
splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)    
for doc in data:
  for chunk in splitter.split_text(doc['content']):
    docs.append(Document(page_content=chunk, metadata={'author': doc['author'], 'created_at': doc['created_at'], 'source': doc['jump_url']}))
    
llm = OpenAI(temperature=0)    
chain = load_summarize_chain(llm, chain_type="map_reduce")
print(chain.run(docs))