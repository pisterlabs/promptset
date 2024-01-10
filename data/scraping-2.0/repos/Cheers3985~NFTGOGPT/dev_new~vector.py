from uuid import uuid4

import tiktoken
from langchain.document_loaders import UnstructuredMarkdownLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 需要魔法
import pinecone
from tqdm import tqdm
from langchain.document_loaders.base import Document
import tempCfg
import re
import os
# token 分词编码
tiktoken.encoding_for_model('gpt-3.5-turbo')
tokenizer = tiktoken.get_encoding('cl100k_base')


os.environ["OPENAI_API_KEY"] = tempCfg.OPENAI_API_KEY
# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

tiktoken_len("hello I am a chunk of text and using the tiktoken_len function "
             "we can find the length of this chunk of text in tokens")
# text 为 md 文档中所有字符串，输出为['','']形式 list[str]
def split_text(text,chunk_size=400,chunk_overlap=20,):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""])
    docs = text_splitter.split_text(text)
    return docs
def insert_vector_store(index_name="nftgo-demo",document: list[Document] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # initialize pinecone
    pinecone.init(
        api_key=tempCfg.PINECONE_API_KEY,  # find at app.pinecone.io
        environment="us-east4-gcp"  # next to api key in console
    )
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536)  # 1536 dim of text-embedding-ada-002
    # 连接至新的index
    index = pinecone.Index(index_name)
    # 每次最多批量加入100个文档内容
    batch_limit = 100

    texts = []
    metadatas = []
    # record 为 Document 类型，下面的代码目的是为了批量进行向量化和存储
    for i, record in enumerate(tqdm(document)):
        # first get metadata fields for this record
        metadata = record.metadata
        # now we create chunks from the record text
        record_texts = record.page_content
        record_texts = split_text(record_texts)  # Document 格式

        # create individual metadata dicts for each chunk,单个文件进行分类[{'chunk':0,'text':'title...','metadata':'},] list[dict]
        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(record_texts)]
        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))] # 使用uuid创建唯一标识符
            embeds = embeddings.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            # 到达200清空处理
            texts = []
            metadatas = []
    # 如果不能被200整除，剩余额外处理
    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embeddings.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds,metadatas)) # 使用uuid创建唯一标识符
    print(index.describe_index_stats())
# print()
# 加载文件夹目录下所有markdown文档的内容，每个文件夹一个Document。
def load_document(markdown_path):
    document_ls = []
    for doc in tqdm(os.listdir(markdown_path)):
        print("Loading Document")
        print("---"+doc + "---")
        if doc.endswith('.md'):
            # try:
                # print(markdown_path+doc)
            loader = UnstructuredMarkdownLoader("docs/"+doc)
            document = loader.load()
            # 加入目录下所有拆分之后的md
            document_ls.extend(document)
    # 加载结束
    print("Loading Finished!")
    # 将document转换成向量数据库
    insert_vector_store(index_name="nftgo-demo",document=document_ls)

            # print(docs)
            # except Exception as e:
            #     print(e)
            #     message = "以下文档未加载成功"
            #     messagel.append(doc)

if __name__ == '__main__':

    load_document(r'D:\Disktop\工作\链坊科技\nftgo\dev_new\docs')
    print('load document to vector database finished')




# docs = split_docs(documents)






'''
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536)  # 1536 dim of text-embedding-ada-002
index = pinecone.GRPCIndex(index_name)
'''
# index.describe_index_stats()
# from tqdm.auto import tqdm
# from uuid import uuid4
#
# batch_limit = 100
#
# texts = []
# metadatas = []
#
# for i, record in enumerate(tqdm(data)):
#     # first get metadata fields for this record
#     metadata = {
#         'wiki-id': str(record['id']),
#         'source': record['url'],
#         'title': record['title']
#     }
#     # now we create chunks from the record text
#     record_texts = text_splitter.split_text(record['text'])
#     # create individual metadata dicts for each chunk
#     record_metadatas = [{
#         "chunk": j, "text": text, **metadata
#     } for j, text in enumerate(record_texts)]
#     # append these to current batches
#     texts.extend(record_texts)
#     metadatas.extend(record_metadatas)
#     # if we have reached the batch_limit we can add texts
#     if len(texts) >= batch_limit:
#         ids = [str(uuid4()) for _ in range(len(texts))]
#         embeds = embed.embed_documents(texts)
#         index.upsert(vectors=zip(ids, embeds, metadatas))
#         texts = []
#         metadatas = []
#
# if len(texts) > 0:
#     ids = [str(uuid4()) for _ in range(len(texts))]
#     embeds = embed.embed_documents(texts)
#     index.upsert(vectors=zip(ids, embeds, metadatas))