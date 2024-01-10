import os
import re
from datetime import datetime
from nosql_database import upload_database
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from tqdm.autonotebook import tqdm

import os

#Converstional retrieval chain

from langchain.text_splitter import CharacterTextSplitter


load_dotenv()

#specify root directory to search for pdfs
root_directory = '/content/drive/MyDrive/projects/gradient_annualreports'

def find_pdfs(directory: str)-> list:
    pdf_list = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                pdf_list.append(os.path.join(root,file))

    return pdf_list

# Usage example
def convert_docs_fordb(pdf_files: list) -> dict:
    directory_path = "/content/drive/MyDrive/projects/gradient_annualreports" #This is the directory your files are
    pdf_files = find_pdfs(directory_path)
    docs_dic = [{'date.uploaded': datetime.today(), 'pdf_file': pdf} for pdf in pdf_files]
    return docs_dic

def chunker(pdf_files):
  annual_reports = []
  for pdf in pdf_files:
      loader = PyPDFLoader(pdf)
      # Load the PDF document
      document = loader.load()
      # Add the loaded document to our list
      annual_reports.append(document)

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

  chunked_annual_reports = []

  for annual_report in annual_reports:
    # Chunk the annual_report
    texts = text_splitter.split_documents(annual_report)
    [print(type(o) for text in texts)]
    # Clean the chunks
    #texts = [re.sub(r'[^a-zA-Z0-9\s]', '', report) for report in texts]
    # Add the chunks to chunked_annual_reports, which is a list of lists
    chunked_annual_reports.append(texts)
    print(chunked_annual_reports)

    return chunked_annual_reports

def doc_handler():
  print("started...")
  pdf_list = find_pdfs(root_directory)
  print(pdf_list)
  print("chunking text...")
  chunked_text = chunker(pdf_list)

  return chunked_text, pdf_list


