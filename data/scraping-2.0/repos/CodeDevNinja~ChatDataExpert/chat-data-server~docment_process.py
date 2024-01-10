from langchain.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader
import os

def doc_load(file_path):
    """
    加载文件Document
    """
    if not os.path.exists(file_path):
        raise Exception('文件不存在')
    file_type = file_path.split('.')[-1]
    if file_type == 'docx':
        doc_load = Docx2txtLoader(file_path)
        docs = doc_load.load()
        return docs
    elif file_type == 'txt':
        txt_load = TextLoader(file_path)
        txt = txt_load.load()
        return txt
    elif file_type == 'pdf':
        pdf_load = PyPDFLoader(file_path)
        pdf = pdf_load.load()
        return pdf
    else:
        raise Exception('文件格式不支持')