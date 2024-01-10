import openai, os
from llama_index import StorageContext, load_index_from_storage
openai.api_key = os.environ.get("OPEN_API_KEY")

#rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir='/Users/zm/aigcData/index_mr_fujino')
#load index
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()
response = query_engine.query("鲁迅先生在日本学习医学的老师是谁")
#response = query_engine.query("孙悟空的师傅是谁")

print(response)