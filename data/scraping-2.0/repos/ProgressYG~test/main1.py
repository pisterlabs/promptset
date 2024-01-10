##
## Chat PDF 만들기
## 2023.10.03
##
## main.py를 이용하여 Streamlit 서비스 만들기
## =>Stream 출력 기능 추가(스트림 옵션)
##

#env 환경변수 읽어 들이기(같은 폴더 "/.env"의 파일을 그대로 읽어옴)
#from dotenv  import load_dotenv
#load_dotenv()

#split 와 chroma DB 간의 충돌 해결
#sqlite3 문제 해결코드 ~~-> pip install pysqlite3-binary 필요
#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#
#pip install chromadb==0.3.29 로 크로마 DB 설치하면 문제해결

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  #텍스트를 페이지 단위보다 더 Split하기
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


from langchain.embeddings import OpenAIEmbeddings

import streamlit as st
import tempfile
import os

###Loader PDF file
#loader = PyPDFLoader("unsu.pdf")
#pages = loader.load_and_split()  #페이별로 쪼개기

##Front page 꾸미기
st.title("Chat Pdf") 
st.write("---")
uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
st.write("---")

##Streamit 에서 파일을 업로드 받기 Function
def pdf_to_document(uploaded_file):
   temp_dir = tempfile.TemporaryDirectory()
   temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
   with open(temp_filepath, "wb") as f:
      f.write(uploaded_file.getvalue())
   loader = PyPDFLoader(temp_filepath)
   pages = loader.load_and_split()
   return pages

#업로드되면 동작하는 코드
if uploaded_file is not None:
   pages = pdf_to_document(uploaded_file)

   #Recursively split by characters 를 사용함
   ###Split 
   text_splitter = RecursiveCharacterTextSplitter(
      # Set a really small chunk size, just to show.
         chunk_size = 300, #자르는 크기
         chunk_overlap  = 20, #split 시 각 단위별 겹침의 크기
         length_function = len,
         is_separator_regex = False,
   )   
   
   texts = text_splitter.split_documents(pages)  #페이지별로 더 쪼개기

   #AIOpenAIEmbeddings 모델을 사용함
   embeddings_model = OpenAIEmbeddings()
   
   # load it into Chroma (랭체인 문서 참조)
   ## python langchain Doc -> Modules -> Vectorstores -> Chroma 참조
   ## Basic Example 에 Disk 저장하는 예제도 있음
   db = Chroma.from_documents(texts, embeddings_model) ## 메모리에 저장하기
   #Disk에 저장하는 예제
   #db = Chroma.from_documents(texts, embeddings_model, persist_directory="./chroma_db")

   #Stream(타타타 글자출력하기) 받아 줄 Hander 만들기
   from langchain.callbacks.base import BaseCallbackHandler
   class StreamHandler(BaseCallbackHandler):
      def __init__(self, container, initial_text=""):
         self.container = container
         self.text=initial_text
      def on_llm_new_token(self, token: str, **kwargs) -> None:
         self.text+=token
         self.container.markdown(self.text)

   #Question
   st.header("PDF에게 질문해 보세요!!")
   question = st.text_input("질문을 입력하세요")

   if st.button("질문하기"):
      with st.spinner('잠시 기다려주세요...!!') :
         #Stream 출력을 위한 Chat Box 만들기(stream_hander)
         chat_box = st.empty()
         stream_hander = StreamHandler(chat_box)

         ## python langchain Doc -> USE cases -> Question Answering 참조
         llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True, callbacks=[stream_hander])
         qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
         qa_chain({"query": question})
   pass



