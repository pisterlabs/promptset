import PyPDF2
import os
from langchain.document_loaders import TextLoader

def pdf_to_txt(pdf, db_path):
    pdf_filename = pdf.name  
    txt_filename = pdf_filename + '.txt'
    txt_path = db_path + txt_filename
    with open(os.path.join(db_path, pdf_filename), 'wb') as pdf_file:
        pdf_file.write(pdf.read())
    with open(os.path.join(db_path, pdf_filename), 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)
    loader = TextLoader(txt_path)
    documents = loader.load()
    return documents
