from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import os

import PyPDF2

pdf_file_obj = open('s12984-021-00942-z.pdf', 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
num_pages = len(pdf_reader.pages)
detected_text = ''

for page_num in range(num_pages):
    page_obj = pdf_reader.pages[page_num]
    detected_text += page_obj.extract_text() + '\n\n'

pdf_file_obj.close()
print(len(detected_text))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.create_documents([detected_text])

directory = 'index_store'
vector_index = FAISS.from_documents(texts, OpenAIEmbeddings())
vector_index.save_local(directory)

vector_index = FAISS.load_local('index_store', OpenAIEmbeddings())
retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k":6})
qa_interface = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo-1106"), 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True,
)

response = qa_interface(
    "What kind of study is this?"
)

print(response["result"])