from langchain.document_loaders import UnstructuredWordDocumentLoader,PyPDFium2Loader,DirectoryLoader,PyPDFLoader,TextLoader
import os

def load_pdf(directory_path):
    data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            print(filename)
            # print the file name
            loader = PyPDFium2Loader(f'{directory_path}/{filename}')
            print(loader)
            data.append(loader.load())
    return data

def load_word(directory_path):
    data = []
    for filename in os.listdir(directory_path):
        # check if the file is a doc or docx file
        # 检查所有doc以及docx后缀的文件
        if filename.endswith(".doc") or filename.endswith(".docx"):
            # langchain自带功能，加载word文档
            loader = UnstructuredWordDocumentLoader(f'{directory_path}/{filename}')
            data.append(loader.load())

    return data


def load_txt(directory_path):
    data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            print(filename)
            loader = TextLoader(f'{directory_path}/{filename}')
            print(loader)
            data.append(loader.load())

    return data