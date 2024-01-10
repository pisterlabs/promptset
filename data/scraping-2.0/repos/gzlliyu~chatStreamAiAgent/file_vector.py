import json
import os
import shutil
from datetime import datetime
from typing import List

import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.chroma import Chroma

from zhipuaiEmbed import ZhipuAiEmbeddings


def ingest_docs(collection='', llm_type: str = 'openai'):
    print('llm_type:', llm_type)
    split_docs = load_qa_xlsx_docs('./assets/示例Q&A.xlsx')

    # 调用函数来删除内容
    delete_old_vector(collection=collection, llm_type=llm_type)

    # 初始化embeddings对象
    embeddings: Embeddings
    if llm_type == 'openai':
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = ZhipuAiEmbeddings()

    # 使用embeddings向量化文档并存入chroma数据库中,并持久化
    print('[', datetime.now(), ']:', "【ingest.py】：向量化进行中......")
    # openai持久化目录
    persist_path = os.path.join(os.getcwd(), 'assets', 'vector_index', 'openai', collection)
    if llm_type == 'zhipuai':
        # zhipuai持久化目录
        persist_path = os.path.join(os.getcwd(), 'assets', 'vector_index', 'zhipuai', collection)

    try:
        docsearch = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_path)
        docsearch.persist()
    except Exception as e:
        print('异常:', e)


def load_qa_xlsx_docs(xlsx_file_path) -> List[Document]:
    """加载xlsx（Excel）文件，生成List[Document]"""

    result = []
    data_frame = pd.read_excel(xlsx_file_path)

    # 选择前2列数据
    selected_columns = data_frame.iloc[:, :2]

    # 将数据转换为 JSON 数组
    json_array = selected_columns.to_dict(orient='records')
    # json_array = json.loads(json_array)
    for item in json_array:
        page_content = json.dumps(item, ensure_ascii=False)
        doc = Document(page_content=page_content)
        result.append(doc)
    return result


def delete_old_vector(collection: str, llm_type: str = 'openai'):
    print('[', datetime.now(), ']:', '开始删向量旧文件....llm_type=', llm_type, 'collection=', collection)
    folder_path = os.path.join(os.getcwd(), 'assets', 'vector_index', 'openai', collection)
    if llm_type == 'zhipuai':
        folder_path = os.path.join(os.getcwd(), 'assets', 'vector_index', 'zhipuai', collection)

    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 递归删除文件夹中的内容
        shutil.rmtree(folder_path)
        print('[', datetime.now(), ']:', f"【ingest.py】：已成功删除 {folder_path} 文件夹中的内容。")
    else:
        print('[', datetime.now(), ']:', f"{folder_path} 文件夹不存在。无需删除内容。")
