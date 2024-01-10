import openai, os
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage

os.environ["OPENAI_API_KEY"] = 'sk-NYsoG3VBKDiTuvdtC969F95aFc4f45379aD3854a93602327'
os.environ["OPENAI_API_BASE"] = 'https://key.wenwen-ai.com/v1'

# 1. 构建索引存储到本地文件
documents = SimpleDirectoryReader('./data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)
index.storage_context.persist('index_data')

# 2. 加载索引
storage_context = StorageContext.from_defaults(persist_dir="./index_data")
index = load_index_from_storage(storage_context)

# 3. 进行查询
query_engine = index.as_query_engine() 
response = query_engine.query("什么是八爪鱼?")
print(response)