import sys
import os
import json
import argparse
import logging
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
from llama_index import LLMPredictor, PromptHelper, ServiceContext
from langchain.chat_models import ChatOpenAI


class UECQueryEngine:
    def __init__(self, reindex):
        with open('./config.json', 'r') as f:
            config = json.load(f)

        os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
        # define prompt helper
        # set maximum input size
        max_input_size = 4096
        # set number of output tokens
        num_output = 256
        # set maximum chunk overlap
        max_chunk_overlap = 20
        prompt_helper = PromptHelper(context_window=max_input_size, num_output=num_output)
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

        if reindex:
            # インデックスの再作成
            documents = SimpleDirectoryReader("./data").load_data()
            index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
            # インデックスの保存
            index.storage_context.persist()
        else:
            # インデックスの読み込み
            storage_context = StorageContext.from_defaults(persist_dir="./storage")
            index = load_index_from_storage(storage_context)

        # クエリエンジンの作成
        self.query_engine = index.as_query_engine()

    def query(self, question):
        return self.query_engine.query(question)

    def make_index(self):
        documents = SimpleDirectoryReader("./data").load_data()
        index = GPTVectorStoreIndex.from_documents(documents)
        # インデックスの保存
        index.storage_context.persist()


if __name__ == "__main__":
    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(description='QA Chatbot for UEC CED')
    parser.add_argument('--reindex', action='store_true', help='インデックスの再作成')
    args = parser.parse_args()

    query_engine = UECQueryEngine(args.reindex)
    question = ""
    while question != "exit":
        question = input("電気通信大学に関する質問を入力してください: ")
        answer = query_engine.query(question)
        print(answer)
