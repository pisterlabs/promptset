# -*- coding: utf-8 -*-
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import os
import PyPDF2
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Path where the PDFs Reside
directory_path = "/Users/schumbar/Desktop/team_projects/CMPE256/CMPE256_Team_Project/flask_app/Data/PDF"
OPENAI_API_KEY = 'XXXX'

def list_filenames(directory):
    try:
        # List all files and directories in the given directory
        filenames = os.listdir(directory)
        # Filter out directories if you want only files
        filenames = [file for file in filenames if os.path.isfile(os.path.join(directory, file))]
        return filenames
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Open the PDF file
def extract_text(file_path):
  pdf_file_path = file_path
  with open(pdf_file_path, 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
  # Create a PDF object
    pdf_text = ''
    for page_num in range(len(pdf_reader.pages)):
      page = pdf_reader.pages[page_num]
      pdf_text += page.extract_text()
  return pdf_text

def extract_text_from_all_PDF_files(file_list):
  output_text = ''
  full_directory_prefix = directory_path
  for each_file_name in file_list:
    output_text = output_text + extract_text(full_directory_prefix + '/' + each_file_name)
  cleaned_text = clean_text(output_text)
  crucial_text = tokenize_and_remove_stopwords(cleaned_text)
  return str(crucial_text)


# Function to clean text data
def clean_text(text):
    """
    Function to clean the text data. This includes:
    - Lowercasing the text
    - Removing special characters and numbers
    - Stripping extra white spaces
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra white spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    return text.strip()

def tokenize_and_remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words]
     
def get_response(question):
  file_list = list_filenames(directory_path)
  all_text = extract_text_from_all_PDF_files(file_list)
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  texts = text_splitter.create_documents([all_text])

  os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
  directory = 'index_store'
  vector_index = FAISS.from_documents(texts, OpenAIEmbeddings())
  vector_index.save_local(directory)

  vector_index = FAISS.load_local('index_store', OpenAIEmbeddings())
  retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k":6})
  qa_interface = RetrievalQA.from_chain_type(llm=ChatOpenAI(),
                                            chain_type="stuff",
                                            retriever=retriever,
                                            return_source_documents=True
                                            )
  response_object = qa_interface(question)
  question = response_object['query']
  response = response_object['result']
  return response


if __name__ == '__main__':
   get_response("What is the difference between a data scientist and a data engineer?")
   print("Done")