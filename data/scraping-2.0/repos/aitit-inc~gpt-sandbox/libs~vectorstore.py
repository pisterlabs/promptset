from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import openai
from langchain.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from .config import ChromaConfig
load_dotenv()


# 定数定義
EMBEDDING_MODEL = "text-embedding-ada-002"

# APIキーの読み込み
openai.api_key = os.getenv("OPENAI_API_KEY")


class VectorStore:
    def __init__(self, api_key, substitution, config, model="gpt-3.5-turbo"):
        self.api_key = api_key
        self.config = ChromaConfig(substitution, config)
        self.llm = OpenAI(model_name=model, openai_api_key=api_key)
        self.embedding = OpenAIEmbeddings(openai_api_key=api_key)

    def create_db(self, texts):
        # テキストデータからベクトルDBを作成
        """
        説明：分割されたテキストをChromaDBに保持
        引数：texts = 分割されたテキスト
            　embedding = 埋め込み表現に使うクラス
            　collection_name = ChromaDBからの参照名
            　persist_directory = 保存先（ディレクトリ）
        """
        db = Chroma.from_documents(texts,
                                   self.embedding,
                                   collection_name=self.config.collection_name(),
                                   persist_directory=self.config.persistant_directory())
        
        # DBをローカルに保存
        db.persist()

        # サーチできるように書き込みデータからロード
        # TODO 前回実行時のデータが残っているので、削除するようにする
        # 現状：環境設定ファイルのpersist_directoryを変更することで対応
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.config.persistant_directory()
        ))
        embeddings = openai.Embedding()
        self.collection = client.get_collection(name=self.config.collection_name(), embedding_function=embeddings)
 
    def retrieve_data(self, vectorstore, query, k=5):
        # ベクトル空間からテキストを探索
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        result = qa({"query": query})
        return result["result"]
    
    def vector_search(self, query, n_results=1):
        """
        説明：ChromaDBから最も関連性の高いテキストデータを抽出
        """
        # クエリをEmbedding化
        query_embed = openai.Embedding.create(
            input=query,
            model=EMBEDDING_MODEL
        )["data"][0]["embedding"]

        query_results = self.collection.query(
            query_embeddings=query_embed,
            n_results=n_results,
            include=["documents"]
        )

        return query_results["documents"][0]