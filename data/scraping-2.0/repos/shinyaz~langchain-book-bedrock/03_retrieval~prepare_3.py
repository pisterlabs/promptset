import sys
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import BedrockEmbeddings
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import Chroma
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

loader = PyMuPDFLoader("./sample.pdf")
documents = loader.load()

text_splitter = SpacyTextSplitter(
    chunk_size=300,
    pipeline="ja_core_news_sm"
)
splitted_documents = text_splitter.split_documents(documents)

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1"
)

database = Chroma(
    persist_directory="./.data",
    embedding_function=embeddings
)

database.add_documents(
    splitted_documents,
)

print("データベースの作成が完了しました。")
