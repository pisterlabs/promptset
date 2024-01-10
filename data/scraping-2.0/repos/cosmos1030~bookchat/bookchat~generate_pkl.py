import os
from django.conf import settings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from .models import Book

os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_KEY') # 환경변수에 OPENAI_API_KEY를 설정합니다.

def pdf_to_data(id): # pdf를 리스트 형태로 변환
    book = Book.objects.get(id=id)
    path = os.path.join(settings.MEDIA_ROOT, book.file.path)
    loader = PyPDFLoader(path)
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    data = text_splitter.split_documents(data)
    return data

def make_vectorstore(id):
    data = pdf_to_data(id)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(data, embeddings)
    return vector_store

def make_pickle(id): # 벡터저장소를 pkl 형태의 파일로 저장
    vectorstore = make_vectorstore(id)
    book = Book.objects.get(id=id)
    pkl_filename = book.title+'.pkl'
    with open(os.path.join(settings.MEDIA_ROOT, 'pickle/', pkl_filename), "wb") as f:
	    pickle.dump(vectorstore, f)
            
def save_pickle(id):
    make_pickle(id)
    book = Book.objects.get(id=id)
    pkl_filename = book.title+'.pkl'
    book.pickle = os.path.join(settings.MEDIA_ROOT, 'pickle', pkl_filename)
    book.save()