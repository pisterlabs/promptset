import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

try:
  API_KEY = open("API_KEY", "r").read()
except FileNotFoundError:
  pass

from get_csv import get_csv
from get_pdf import get_pdf

csv_PATH = ["CSV_Base/bathroom_location.csv",
         "CSV_Base/export_stands20230922.csv",
          "CSV_Base/FDV_TRANSN_updated.csv"]
pdf_PATH = "CSV_Base/FAQ_FDV_Zenko_V1.2.pdf"

texts_col = []
for PATH in csv_PATH:
  texts_col.append(get_csv(PATH))
text_PDF = get_pdf(pdf_PATH)

texts_col.append(text_PDF)


# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

## here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = OpenAIEmbeddings(openai_api_key=API_KEY)

for i in range(len(texts_col)):
  vectordb = Chroma.from_texts(texts=texts_col[0], 
                                 embedding=embedding,
                                 persist_directory=persist_directory) 
                                 
vectordb = Chroma.from_documents(documents=texts_col[i], 
                                 embedding=embedding,
                                  persist_directory=persist_directory)

                                

# persiste the db to disk
vectordb.persist()
vectordb = None

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

retriever = vectordb.as_retriever()
docs = retriever.get_relevant_documents("How much money did Pando raise?")
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# create the chain to answer questions 
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=API_KEY), 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)

## Cite sources
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
