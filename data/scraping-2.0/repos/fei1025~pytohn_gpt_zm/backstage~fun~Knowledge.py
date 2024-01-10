import hashlib
import logging
import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from sqlalchemy.orm import Session

from langchain.document_loaders import PyPDFLoader
from entity import models, crud
import pandas as pd
from langchain.schema import Document


# 创建一个知识库
def create_knowledge(knowledge: models.knowledge, db: Session):
    setting = crud.get_user_setting(db)

    index_path = "./index/paper"
    embeddings = OpenAIEmbeddings(openai_api_base=setting.openai_api_base,
                                  openai_api_key=setting.openai_api_key)
    file_src = knowledge.file_path
    documents = get_documents(file_src)
    index = FAISS.from_documents(documents, embeddings)
    # os.makedirs("./index", exist_ok=True)
    index_name = string_to_md5(file_src)
    index_path = index_path + "/" + index_name
    index.save_local(index_path)
    knowledge.index_name = index_name
    knowledge.index_path = index_path
    crud.save_knowledge(db, knowledge)



def add_knowledge(knowledge: models.knowledge, db: Session):
    index_path = knowledge.index_path
    setting = crud.get_user_setting(db)
    embeddings = OpenAIEmbeddings(openai_api_base=setting.openai_api_base, openai_api_key=setting.openai_api_key)
    documents = get_documents(knowledge.file_path)
    index = FAISS.load_local(index_path, embeddings)
    index.aadd_documents(documents)
    index.save_local(index_path)


def get_knowledge(knowledge: models.knowledge, db: Session) -> FAISS:
    load_knowledge = crud.get_knowledge(db, knowledge)
    index_path = load_knowledge.index_path
    setting = crud.get_user_setting(db)
    embeddings = OpenAIEmbeddings(openai_api_base=setting.openai_api_base, openai_api_key=setting.openai_api_key)
    return FAISS.load_local(index_path, embeddings)


def get_documents(filepath) -> []:
    documents = []
    from langchain.text_splitter import TokenTextSplitter
    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=30)
    filename = os.path.basename(filepath)
    file_type = os.path.splitext(filename)[1]
    texts = None
    try:
        if file_type == ".pdf":
            loader = PyPDFLoader(filepath)
            texts = loader.load()
        elif file_type == ".docx":
            logging.debug("Loading Word...")
            from langchain.document_loaders import UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(filepath)
            texts = loader.load()
        elif file_type == ".pptx":
            logging.debug("Loading PowerPoint...")
            from langchain.document_loaders import UnstructuredPowerPointLoader
            loader = UnstructuredPowerPointLoader(filepath)
            texts = loader.load()
        elif file_type == ".epub":
            logging.debug("Loading EPUB...")
            from langchain.document_loaders import UnstructuredEPubLoader
            loader = UnstructuredEPubLoader(filepath)
            texts = loader.load()
        elif file_type == ".xlsx":
            logging.debug("Loading Excel...")
            text_list = excel_to_string(filepath)
            texts = []
            for elem in text_list:
                texts.append(Document(page_content=elem,
                                      metadata={"source": filepath}))
        else:
            logging.debug("Loading text file...")
            from langchain.document_loaders import TextLoader
            loader = TextLoader(filepath, "utf8")
            texts = loader.load()
    except Exception as e:
        import traceback
        logging.error(f"Error loading file: {filename}")
        traceback.print_exc()
    if texts is not None:
        texts = text_splitter.split_documents(texts)
        documents.extend(texts)
    return documents


def string_to_md5(string):
    # 创建 MD5 对象
    md5_hash = hashlib.md5()
    # 更新对象中的字符串
    md5_hash.update(string.encode('utf-8'))
    # 获取 MD5 哈希值
    md5_value = md5_hash.hexdigest()
    return md5_value


def excel_to_string(file_path):
    # 读取Excel文件中的所有工作表
    excel_file = pd.read_excel(file_path, engine='openpyxl', sheet_name=None)

    # 初始化结果字符串
    result = []

    # 遍历每一个工作表
    for sheet_name, sheet_data in excel_file.items():
        # 处理当前工作表并添加到结果字符串
        result += sheet_to_string(sheet_data, sheet_name=sheet_name)

    return result


def sheet_to_string(sheet, sheet_name=None):
    result = []
    for index, row in sheet.iterrows():
        row_string = ""
        for column in sheet.columns:
            row_string += f"{column}: {row[column]}, "
        row_string = row_string.rstrip(", ")
        row_string += "."
        result.append(row_string)
    return result
