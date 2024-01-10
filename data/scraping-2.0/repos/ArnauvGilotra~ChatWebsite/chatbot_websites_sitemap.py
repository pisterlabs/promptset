
import os
os.environ["OPENAI_API_KEY"] = "REDACTED"

import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
import nest_asyncio
import sys

nest_asyncio.apply()

from langchain.document_loaders.sitemap import SitemapLoader



sitemap_loader = SitemapLoader( "https://www.buyerforesight.com/sitemap_index.xml" , filter_urls=["https://www.buyerforesight.com/careers", "https://www.buyerforesight.com/not-hiring", "https://www.buyerforesight.com/careers/director-people-culture/"])

sitemap_loader.requests_per_second = 1
# Optional: avoid `[SSL: CERTIFICATE_VERIFY_FAILED]` issue
sitemap_loader.requests_kwargs = {"verify": False}


embeddings = OpenAIEmbeddings()
if os.path.exists("faiss_store_openai.pkl"):
    VectorStore = None
    with open("faiss_store_openai.pkl", "rb") as f:
       VectorStore = pickle.load(f)
    llm=OpenAI()
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())
    qd = {}
    while True:
        inp = input("Enter your query about this site: ")
        qd['question']=inp
        print("Finding Answer:\n")
        #print(chain({"question": "what buyreforesight.com is about??"}, return_only_outputs=True)['answer'])
        print(chain(qd, return_only_outputs=True)['answer'])
        print(chain(qd, return_only_outputs=True)['sources'])
    sys.exit(0)


data = sitemap_loader.load()

text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)

docs = text_splitter.split_documents(data)

print(len(docs))

print("Creating vectorstore..\n")
VectorStore = FAISS.from_documents(docs, embeddings)
print("Saving vectorstore..\n")
with open("faiss_store_openai.pkl", "wb") as f:
  pickle.dump(VectorStore, f)


print("Establishing pipeline (over llm)....\n")

llm=OpenAI()
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())
#resp = chain({"question": "what buyreforesight.com is about??"}, return_only_outputs=True)
print(chain({"question": "what buyreforesight.com is about??"}, return_only_outputs=True)['answer'])
