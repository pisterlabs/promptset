import json
import os
import openai

from dotenv import load_dotenv
from langchain import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage

def main():
    # APIkeyの設定
    load_dotenv()
    try:
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    except KeyError:
        print("OPENAI_API_KEY environment variable not found. Please make sure it is set.")
    openai.api_key = OPENAI_API_KEY

    # モデルの読み込み
    if(0):
        documents = SimpleDirectoryReader(input_dir="./chat_app/data").load_data()
        print("documents: ", documents)
        index = VectorStoreIndex.from_documents(documents)
        # 保存
        index.storage_context.persist()
    else:
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir='./storage')
        # load index
        index = load_index_from_storage(storage_context)

    # クエリの実行
    query_engine = index.as_query_engine()
    response = query_engine.query("Dockerイメージの取得途中で止まる原因は?")
    print("response: ", response)

if __name__ == "__main__":
    main()
