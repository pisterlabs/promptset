# パッケージのインポート
import os
from dotenv import load_dotenv
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage
import langchain
import logging
import sys
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
import tiktoken


def embedding(input_dir, output_dir):
    # 環境変数の設定
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # ログレベルの設定
    # langchain.verbose = True
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # データの読み込み
    documents = SimpleDirectoryReader(input_dir).load_data()

    text_splitter = TokenTextSplitter(separator="。", chunk_size=128, chunk_overlap=0,tokenizer=tiktoken.get_encoding("cl100k_base").encode)

    node_parser = SimpleNodeParser(text_splitter=text_splitter)

    service_context = ServiceContext.from_defaults(node_parser=node_parser)

    # インデックスの作成(Embedding)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
    # インデックスの保存
    index.storage_context.persist(persist_dir=output_dir)


if __name__ == "__main__":
    # 動かすときこれでやる python ./process_data/embedding.py
    input_dir = "./res/for_embedding/ファーストリテイリング/Sciences"
    output_dir = "./res/embedding/ファーストリテイリング/Sciences"
    embedding(input_dir, output_dir)
