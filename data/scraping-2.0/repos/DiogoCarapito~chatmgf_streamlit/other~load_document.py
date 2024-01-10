"""from langchain.document_loaders import UnstructuredPDFLoader
#, OnlinePDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter

file = 'Programa-Nacional-Vigilancia-Gravidez-Baixo-Risco-2015.pdf'
#file = OnlinePDFLoader('http://nocs.pt/wp-content/uploads/2016/01/Programa-Nacional-Vigilancia-Gravidez-Baixo-Risco-2015.pdf')
loader = UnstructuredPDFLoader(file)

data = loader.load()

print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your document')

'''import os
import openai
openai_api_key = os.environ.get('OPENAI_API_KEY')'''

"""

!pip install "unstructured[local-inference]"
!pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2"
!pip install layoutparser[layoutmodels,tesseract]
!pip install langchain
!pip install pinecone
!pip install openai

import pinecone
import openai
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

loader = UnstructuredFileLoader("../data/PNGBR.txt")

docs = loader.load()

print (f'You have {len(docs)} document(s) in your data')
print (f'There are {len(docs[0].page_content)} characters in your document')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

print (f'Now you have {len(texts)} documents')