import logging
import bson
import json
from time import time
from uuid import uuid4
from copy import deepcopy
from datetime import datetime
from langchain.schema import Document, BaseRetriever
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from elasticsearch_dsl import (
    UpdateByQuery,
    Search,
    Q,
    Boolean,
    Date,
    Integer,
    Document,
    InnerDoc,
    Join,
    Keyword,
    Long,
    Nested,
    Object,
    Text,
    connections,
    DenseVector
)
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import get_openai_callback
from app import app

class ObjID():
    def new_id():
        return str(bson.ObjectId())
    def is_valid(value):
        return bson.ObjectId.is_valid(value)


class NotFound(Exception): pass


connections.create_connection(hosts="http://192.168.110.34:9200", basic_auth=('elastic', 'fsMxQANdq1aZylypQWZD'))


class User(Document):
    openid = Keyword()
    name = Text(fields={"keyword": Keyword()})
    status = Integer()
    extra = Object()    # 用于保存 JSON 数据
    created = Date()
    modified = Date()

    class Index:
        name = 'user'


class Collection(Document):
    user_id = Keyword()  # 将字符串作为文档的 ID 存储
    name = Text(analyzer='ik_max_word')
    description = Text(analyzer='ik_max_word')  #知识库描述
    summary = Text(analyzer='ik_max_word')  #知识库总结
    status = Integer()
    created = Date()
    modified = Date()      #

    class Index:
        name = 'collection'

#Documents区别于固有的Docunment
class Documents(Document):
    uniqid = Long()     #唯一性id,去重用
    collection_id = Keyword()  # 将字符串作为文档的 ID 存储
    type = Keyword()    #文档类型用keyword保证不分词
    path = Keyword()    #文档所在路径
    name = Text(analyzer='ik_max_word')
    chunks = Integer() #文档分片个数
    summary = Text(analyzer='ik_max_word')  #文档摘要
    status = Integer()
    created = Date()
    modified = Date()  # 用于保存最后一次修改的时间

    class Index:
        name = 'document'


class Embedding(Document):
    document_id = Keyword()     #文件ID
    collection_id = Keyword()    #知识库ID
    chunk_index = Keyword()    #文件分片索引
    chunk_size = Integer()  #文件分片大小
    document = Text(analyzer='ik_max_word')       #分片内容
    embedding = DenseVector(dims=768)
    status = Integer()
    created = Date()
    modified = Date()  # 用于保存最后一次修改的时间

    class Index:
        name = 'embedding'

class Bot(Document):
    user_id = Keyword()  # 用户ID
    collection_id = Keyword()  # 知识库ID
    hash = Integer()    #hash
    extra = Object()    #机器人配置信息
    status = Integer()
    created = Date()
    modified = Date()  # 用于保存最后一次修改的时间

    class Index:
        name = 'bot'


def init():
    User.init()
    Collection.init()
    Documents.init()
    Collection.init()
    Bot.init()

def get_user(user_id):
    user = User.get(id=user_id)
    if not user:
        raise NotFound()
    return user


def save_user(openid='', name='', **kwargs):
    s = Search(index="user").filter("term", status=0).filter("term", openid=openid)
    response = s.execute()
    if not response.hits.total.value :
        user = User(
            meta={'id': ObjID.new_id()},
            openid=openid,
            name=name,
            status = 0,
            extra=kwargs,
        )
        user.save()
        return user
    else:
        user = User.get(id=response.hits[0].meta.id)
        user.update(openid=openid, name=name, extra=kwargs)
        return user

'''class CollectionWithDocumentCount(Collection):
    s = Documents.search(using="es",index="document").filter("term",collection_id = Collection.meta.id).filter("term", status=0)
    response = s.execute()
    document_count = response.hits.total.value'''

def get_collections(user_id):
    s = Search(index="collection").filter("term", user_id=user_id)
    # 执行查询
    response = s.execute()
    total = response.hits.total.value
    # 返回搜索结果（文档实例的列表）
    if total == 0:
        return  [],0
    return list(response), total


def get_collection_by_id(user_id, collection_id):
    collection = Collection.get(id=collection_id)
    if collection :
        if collection.user_id == user_id:
            return collection
        else:
            return None
    else:
        return collection


def save_collection(user_id, name, description):
    collection_id = ObjID.new_id()
    collection = Collection(
        meta={'id': collection_id},
        user_id=user_id,
        name=name,
        description=description,
        summary='',
    )
    collection.save()
    return collection


def update_collection_by_id(user_id, collection_id, name, description):
    collection = get_collection_by_id(user_id, collection_id)
    if not collection:
        raise NotFound('collection not found')
    collection.name = name
    collection.description = description
    collection.save()


def delete_collection_by_id(user_id, collection_id):
    collection = get_collection_by_id(user_id, collection_id)
    if not collection:
        raise NotFound('collection not found')
    collection.status = -1
    collection.save()
    bots = Search(index="bot").filter("term", collection_id=collection_id).execute()
    for bot in bots:
        bot.update(collection_id='')


def get_document_id_by_uniqid(collection_id, uniqid):
    s = Search(index="document").filter(
        "term",
        collection_id=collection_id,
        uniqid=uniqid,
        status=0
    )
    response = s.execute()
    return list(response)


def get_documents_by_collection_id(user_id, collection_id):
    collection = get_collection_by_id(user_id, collection_id)
    assert collection, '找不到对应知识库'
    s = Search(index="document").filter("term", collection_id=collection_id, status=0).sort({"created": {"order": "desc"}})
    response = s.execute()
    total = response.hits.total.value
    # 返回搜索结果（文档实例的列表）
    if total == 0:
        return [], 0
    return list(response), total


def remove_document_by_id(user_id, collection_id, document_id):
    collection = get_collection_by_id(user_id, collection_id)
    assert collection, '找不到对应知识库'
    doc = Documents.get(id=document_id)
    if doc:
        doc.update(status=-1)
        embeddings = Search(index='embedding').filter("term", document_id=document_id).execute()
        for embedding in embeddings:
            embedding.update(status=-1)


def purge_document_by_id(document_id):
    doc = Documents.get(id=document_id)
    if doc:
        doc.delete()
        embeddings = Search(index='embedding').filter("term", document_id=document_id).execute()
        for embedding in embeddings:
            embedding.delete()


def set_document_summary(document_id, summary):
    doc = Documents.get(id=document_id)
    if doc:
        doc.update(summary=summary)


def get_document_by_id(document_id):
    doc = Documents.get(id=document_id)
    return doc if doc else None


def save_document(collection_id, name, url, chunks, type, uniqid=None):
    did = ObjID.new_id()
    doc = Documents(
        id=did,
        collection_id=collection_id,
        type=type,
        name=name,
        path=url,
        chunks=chunks,
        uniqid=uniqid,
        summary='',
    )
    doc.save()
    return doc

def save_embedding(collection_id, document_id, chunk_index, chunk_size, document, embedding):
    eid = ObjID.new_id()
    embedding = Embedding(
        id=eid,
        collection_id=collection_id,
        document_id=document_id,
        chunk_index=chunk_index,
        chunk_size=chunk_size,
        document=document,
        embedding=embedding,
    )
    embedding.save()
    return embedding


def get_bot_list(user_id, collection_id):
    s = Search(index="bot").filter(
        "term",
        user_id=user_id,
        collection_id=collection_id,
    ).filter(
        "terms",
        status=[0, 1]
    ).sort({"created": {"order": "desc"}})
    response = s.execute()
    total = response.hits.total.value
    # 返回搜索结果（文档实例的列表）
    if total == 0:
        return [], 0
    return list(response), total



def get_bot_by_hash(hash):
    bot = Search(index="bot").filter(
        "term",
        hash=hash,
    ).execute()
    if bot.hits.total.value == 0:
        raise NotFound()
    return bot[0]



def get_bot_by_hash(hash):
    bot = Search(index="bot").filter(
        "term",
        hash=hash,
    ).execute()
    if bot.hits.total.value == 0:
        raise NotFound()
    return bot[0]

def query_by_collection_id(collection_id, q):
    from tasks import embed_query
    embed = embed_query(q)
    query = Q({
  "query": {
    "match": {
      "title": {
        "query": q,
        "boost": 0.00001
      }
    }
  },
  "knn": {
    "field": "content_vector",
    "query_vector":embed ,
    "k": 10,
    "num_candidates": 50,
    "boost": 1.0
  },
  "size": 20,
  "explain": True,
  "_source" : "title"
}
  )
    s = Search(index="embedding").query(query).filter(
        "term",
        collection_id=collection_id,
        status = 0,
    )
    result = s.execute()
    return result
'''    column = Embedding.embedding.cosine_distance(embed)
    query = db.session.query(
        EmbeddingWithDocument,
        column.label('distinct'),
    ).filter(
        Embedding.collection_id == collection_id,
        Embedding.status == 0,
    ).order_by(
        column,
    )
    total = query.count()
    if total == 0:
        return [], 0
    return query_one_page(query, page, size), total'''