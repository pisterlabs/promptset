from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chains import AnalyzeDocumentChain
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI


import pandas as pd

import requests

from .database import add_document

def get_pdf_text(document_path, start_page=1, final_page=999):
    reader = PdfReader(document_path)
    # reader = UnstructuredPDFLoader(document_path)
    number_of_pages = len(reader.pages)
    
    page = ""
    for page_num in range(start_page - 1, min(number_of_pages, final_page)):
        page += reader.pages[page_num].extract_text()
    return page

def summarize(pages):
    model = OpenAI(temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'))
    summary_chain = load_summarize_chain(llm=model, chain_type='map_reduce')
    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
    summary = summarize_document_chain.run(pages)
    return summary

def create_dataframe(title, identifier, author, summary):
    data = {'Id': [identifier], 'Title': [title],'Author': [author] ,'Summary': [summary]}
    df = pd.DataFrame(data)
    return df


def processing(document_path, author):
    pages = get_pdf_text(document_path)
    summary = summarize(pages)
    df = create_dataframe(document_path, author, summary)
    return df

def extract(document_path, author):
    pages = get_pdf_text(document_path)
    df = create_dataframe(document_path, author, pages)
    return df

def read_from_url(url, author,identifier, namespace):
    response = requests.get(url)
    response.raise_for_status()

    with open('temp.pdf', 'wb') as file:
        file.write(response.content)
    
    pages = get_pdf_text('temp.pdf')
    add_document(document_id = identifier, document_file=pages, namespace_name=namespace)
    df = create_dataframe(url, identifier, author, pages)
    os.remove('temp.pdf')
    
    return df

def read_from_encode(file, author, identifier, namespace, title, session_id):
    pdf_reader = PdfReader(file)

    # Get the number of pages in the PDF
    num_pages = len(pdf_reader.pages)
    pages = ""
    # Read the contents of each page
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        pages += text
    new_id = add_document(document_id=identifier, document_title=title , document_author=author, document_file=pages, namespace_name=namespace, session_id=session_id)
    identifier = new_id if new_id is not None else identifier
    df = create_dataframe(title, identifier, author, pages)
    return df, identifier





