from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader, UnstructuredWordDocumentLoader, MWDumpLoader, UnstructuredPowerPointLoader, CSVLoader
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from docx import Document
from ebooklib import epub
import requests
from bs4 import BeautifulSoup
import faiss
from typing import List
import re
from youtube_transcript_api import YouTubeTranscriptApi
from params.config import APIKeyManager
dirname = os.path.dirname("/home/t4u/Desktop/suck/synergi/core/")
static_dir=os.path.join(dirname, "static")
def  docs_chat_tool(prompt,path,types):
       if(types=="youtube"):
            bot = VideoChatbot()
            return bot.QA_video(url,'')
       elif(types=="document"):
            bot=DocumentChatbot()
            return bot.predict_QA(prompt,f"{static_dir}/{path}")
       elif(types=="website"):
            bot=WebsiteChat()
            return QA_website()
       else:
           print('no type specify')

class DocumentChatbot:
    def __init__(self,vector_index_directory='index_store'):
        # Configuration de l'API OpenAI
        os.environ["OPENAI_API_KEY"] =APIKeyManager().get_api_key('openai_key') 
        # Initialisation des composants LangChain
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vector_index_directory = vector_index_directory
        self.vector_index = None
        self.retriever = None
        self.qa_interface = None
 
    def init_retriever(self):
        # Configuration du récupérateur
        retriever = self.vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        return retriever

    def init_qa_interface(self,prompt):
        # Configuration de l'interface de questions-réponses
        qa_interface = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(),
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
        )
        return qa_interface(prompt)

    def predict_QA(self, prompt, document_path):
        # Chargement du texte à partir du document
        document_text = self.load_text_from_document(document_path)
        
        # Recherche de questions-réponses basée sur le prompt et le texte du document
        #response = self.qa_interface(
          #  f"{document_text}\nPrompt: {prompt}"
        #)
        document_text=self.split_text(document_text)
        print(document_text)
        self.create_vectors(document_text)
        self.retrieve_vectors()
        print(self.retriever)
        pkj=self.init_qa_interface(prompt)["result"]
        print(pkj)
        return pkj
        #return response["result"]

    def load_text_from_document(self, document_path):
        file_extension = os.path.splitext(document_path)[-1].lower()
        
        if file_extension == ".pdf":
            text = self.extract_text_from_pdf(document_path)
        elif file_extension == ".doc" or file_extension == ".docx":
            text = self.extract_text_from_docx(document_path)
        elif file_extension == ".epub":
            text = self.extract_text_from_epub(document_path)
        elif file_extension == ".txt":
            text = self.extract_text_from_txt(document_path)
        else:
            raise ValueError("Format de fichier non pris en charge")
        
        return text

    def extract_text_from_pdf(self, document_path):
        text = ""
        pdf = PdfReader(open(document_path, "rb"))
        for page_num in range(len(pdf.pages)):  # Use len(pdf.pages) to get the number of pages
          page = pdf.pages[page_num]
          text += page.extract_text()
        return text

    def extract_text_from_docx(self, document_path):
        doc = Document(document_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def extract_text_from_epub(self, document_path):
        book = epub.read_epub(document_path)
        text = ""
        for item in book.items:
            if isinstance(item, epub.EpubHtml):
                text += item.content
        return text

    def extract_text_from_txt(self, document_path):
        with open(document_path, "r", encoding="utf-8") as file:
            text = file.read()
        return text
    def split_text(self,text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.create_documents([text])
        return texts
    def create_vectors(self,text):
        directory = 'index_store'
        vector_index = FAISS.from_documents(text, OpenAIEmbeddings())
        vector_index.save_local(directory)
    def retrieve_vectors(self):
        vector_index = FAISS.load_local("index_store", OpenAIEmbeddings())
        self.retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 6})

class VideoChatbot:
      def __init__(self):
         print('lol')

      def  QA_video(self,path,types):
          if(types=='local'):
            print('lol2')
          elif(types=='web'):
            print('lol3')
          else:
             #get the id of the youtube video first 
             def get_video_id(url):
                youtube_url_pattern = r'(https?://)?(www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)'
                youtube_short_url_pattern = r'(https?://)?(www\.)?youtu\.be/([a-zA-Z0-9_-]+)'
                match = re.search(youtube_url_pattern, url) or re.search(youtube_short_url_pattern, url)
                if match:
                      video_id = match.group(3)
                      print(f"Video ID: {video_id}")
                      return video_id
                else:
                   print(f"Invalid YouTube URL: {url}")
             
             
             print(YouTubeTranscriptApi.get_transcript('jd0TyMFnkOM',languages=['fr', 'en']))
             
 
class WebsiteChat:
      def __init__(self):
        self.retriever=''
        print("chat with website")
      def  QA_website(self,url,prompt):
                try:
                   response = requests.get(url)
                   if response.status_code == 200:
                      soup = BeautifulSoup(response.text, 'html.parser')
                      text_elements = soup.find_all(text=True)
                      website_text = ' '.join(text.strip() for text in text_elements if text.strip())
                      website_text=self.split_text(website_text)
                      self.create_vectors(website_text)
                      self.retrieve_vectors()
                      print(self.init_qa_interface(prompt))
                   else:
                       print(f"Failed to retrieve content. Status code: {response.status_code}")
                except Exception as e:
                   print(f"An error occurred: {str(e)}")
                   return None
      def split_text(self,text):
          text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
          texts = text_splitter.create_documents([text])
          return texts
      def create_vectors(self,text):
          directory = 'index_store2'
          vector_index = FAISS.from_documents(text, OpenAIEmbeddings())
          vector_index.save_local(directory)
      def retrieve_vectors(self):
          vector_index = FAISS.load_local("index_store2", OpenAIEmbeddings())
          self.retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 6})
      def init_qa_interface(self,prompt):
        # Configuration de l'interface de questions-réponses
        qa_interface = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(),
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
        )
        return qa_interface(prompt)

