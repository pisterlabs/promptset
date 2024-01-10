# -*- coding: utf-8 -*-
import os
import re
import openai
import glob
import importlib
from traceback import print_exc
from loguru import logger
from typing import List
from langchain.schema import Document

EMBEDDING_URL = os.getenv("EMBEDDING_URL")
STATIC_ROOT_PATH = os.getenv("STATIC_ROOT_PATH")
KB_ROOT_PATH = os.getenv("KB_ROOT_PATH")



LOADER_DICT = {
    "UnstructuredFileLoader": [
        '.eml', '.html', '.json', '.md', '.msg', '.rst',
        '.rtf', '.txt', '.xml', '.doc', '.docx', '.epub',
        '.odt', '.pdf', '.ppt', '.pptx', '.tsv'
    ],
    "CSVLoader": [".csv"],
    "PyPDFLoader": [".pdf"],
}

SUPPORTED_EXTS = set([ext for sublist in LOADER_DICT.values() for ext in sublist])


def get_loader_class(file_extension):
    for c, exts in LOADER_DICT.items():
        if file_extension in exts:
            return c


def remove_extra_spaces_and_newlines(text):
    # 替换两个及以上的空格为一个空格
    text = re.sub(' +', ' ', text)
    # 替换两个及以上的换行符为一个换行符
    text = re.sub('\n+', '\n', text)
    return text


def cut_chinese_sent(para):
    """
    Cut the Chinese sentences more precisely, reference to "https://blog.csdn.net/blmoistawinde/article/details/82379256".
    """
    para = re.sub(r"([。！？\?])([^”’])", r"\1\n\2", para)
    para = re.sub(r"(\.{6})([^”’])", r"\1\n\2", para)
    para = re.sub(r"(\…{2})([^”’])", r"\1\n\2", para)
    para = re.sub(r"([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
    para = para.rstrip()
    return para.split("\n")


def extract_qa_pairs(text):
    """
    Extract question-answer pairs from the text and return the pairs and the remaining text.
    """
    # pairs = re.findall(r'(问题[^\n]*：[^\n]*\n答案：[^\n]*)', text)
    pairs = re.findall(r'#custom_QA#\n(.*?)\n#custom_QA#', text, re.DOTALL)
    for pair in pairs:
        text = text.replace(pair, '')
    return pairs, text


def auto_splitter(input_text, max_text_len=600):
    sens = []
    _sens = cut_chinese_sent(input_text)
    tmp = ""
    for s in _sens:
        if len(tmp + s) > max_text_len:
            sens.append(tmp)
            tmp = ""
        tmp += s
    if tmp:
        sens.append(tmp)
    return sens


import traceback


def load_document(filepath, chunk_size: int = 600, chunk_overlap: int = 10):
    """ 加载文档 """
    ext = os.path.splitext(filepath)[-1].lower()
    document_loader_name = get_loader_class(ext)
    try:
        document_loaders_module = importlib.import_module('langchain.document_loaders')
        DocumentLoader = getattr(document_loaders_module, document_loader_name)
    except Exception as e:
        traceback.print_exc()
        document_loaders_module = importlib.import_module('langchain.document_loaders')
        DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")

    if document_loader_name == "UnstructuredFileLoader":
        loader = DocumentLoader(filepath, autodetect_encoding=True)

        # 此处增加一个自定义占位符，直接根据占位符切分。
        pairs, _ = extract_qa_pairs(loader.load()[0].page_content)
        if pairs:
            return [Document(page_content=remove_extra_spaces_and_newlines(pair), metadata={"source": filepath}) for
                    pair in pairs]

    else:
        loader = DocumentLoader(filepath)
        # 此处增加一个自定义占位符，直接根据占位符切分。
        pairs, _ = extract_qa_pairs(loader.load()[0].page_content)
        if pairs:
            return [Document(page_content=remove_extra_spaces_and_newlines(pair), metadata={"source": filepath}) for
                    pair in pairs]

    try:
        text_splitter_module = importlib.import_module('langchain.text_splitter')
        TextSplitter = getattr(text_splitter_module, "SpacyTextSplitter")
        text_splitter = TextSplitter(
            pipeline="zh_core_web_sm",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except Exception as e:
        traceback.print_exc()
        text_splitter_module = importlib.import_module('langchain.text_splitter')
        TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")
        text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    loader = [Document(page_content=remove_extra_spaces_and_newlines(loader.load()[0].page_content),
                       metadata=loader.load()[0].metadata)]

    # docs = loader.load_and_split(text_splitter)

    try:
        docs = text_splitter.split_documents(loader)
        del loader
    except Exception as e:
        del docs
        traceback.print_exc()
        max_length = 1000000  # Spacy's default max length
        pre_docs = [
            [Document(page_content=loader[0].page_content[:max_length], metadata=loader[0].metadata)],
            [Document(page_content=loader[0].page_content[max_length:], metadata=loader[0].metadata)],
        ]
        docs = []
        for item in pre_docs:
            docs.extend(text_splitter.split_documents(item))
        del pre_docs
        del loader
    docs_clean = []

    try:
        # 使用正则清除多余的\n、空格等
        docs = [Document(page_content=remove_extra_spaces_and_newlines(item.page_content), metadata=item.metadata) for
                item in docs]

        # 强制切分超出长度的段落
        for item in docs:
            LEN = len(item.page_content)
            if LEN <= chunk_size:
                docs_clean.append(item)
            else:
                docs_list = auto_splitter(item.page_content, max_text_len=chunk_size)
                docs_clean.extend([Document(page_content=doc, metadata=item.metadata) for doc in docs_list])
    except Exception as e:
        traceback.print_exc()

    return docs_clean


class Embeddings:
    def __init__(self):
        self.client = openai.Embedding()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        openai.api_key = 'xxxx'
        openai.api_base = EMBEDDING_URL
        all_embedding = []
        for i in range(0, len(texts), 1000):
            embeddings = self.client.create(input=texts[i:i + 1000], model="text2vec-large-chinese")
            all_embedding.extend(([e['embedding'] for e in embeddings['data']]))
        return all_embedding

    def embed_query(self, text: str) -> List[float]:
        openai.api_key = 'xxxx'
        openai.api_base = EMBEDDING_URL
        return self.client.create(input=text, model="text2vec-large-chinese")['data'][0]['embedding']


def validate_kb_name(knowledge_base_id: str) -> bool:
    # 检查是否包含预期外的字符或路径攻击关键字
    if "../" in knowledge_base_id:
        return False
    return True


def get_kb_path():
    return os.path.join(KB_ROOT_PATH)


def get_file_path(doc_name: str):
    return os.path.join(STATIC_ROOT_PATH, doc_name)


def list_kbs_from_folder(path=STATIC_ROOT_PATH):
    return [f for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f))]


def list_files_from_folder():
    return [file for file in os.listdir(STATIC_ROOT_PATH)
            if os.path.isfile(os.path.join(STATIC_ROOT_PATH, file))]


def files_list(kb_name: str, file_types=None):
    if file_types is None:
        file_types = ['txt', 'pdf', 'docx', 'json', 'md']
    data = []
    for file_type in file_types:
        search_pattern = os.path.join(os.path.join(os.path.join(KB_ROOT_PATH, kb_name), 'content'), f'*.{file_type}')
        files = glob.glob(search_pattern)

        for file in files:
            file_name = os.path.basename(file)
            data.append(file_name)
    return data


def loadtxt():
    labels = []
    contents = []
    background = []
    with open('./static/source/a.txt', 'r', encoding='utf8') as file:
        for line in file:
            content = line.strip('\n').split('\t')
            if content[1] == '表现形式':
                background.append(content[0] + '的' + content[1] + '为' + content[2])
                labels.append(content[0])
            contents.append(content)

    return contents, labels, background


if __name__ == '__main__':
    # for root, dirs, files in os.walk(STATIC_ROOT_PATH):
    #     for file in files:
    #         if file == '智能问答.docx':
    #             file_path = os.path.join(root, file)
    #             res = load_document(file_path, file)
    #             print(res)
    contents, labels, background = loadtxt()
    print(labels[:10])
    print((background[:10]))
