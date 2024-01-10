from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import os

from app.core.config import settings

embeddings = OpenAIEmbeddings()   

def main() -> None:
    file_path = str(
        os.path.abspath(
        os.path.join(
        __file__,
        '..','lVPbJhbZyy5uI8hHOpZhm.csv')))

    df_docs = pd.read_csv(file_path)
    df_docs = df_docs.dropna(subset=["response_text"])
    
    documents = DataFrameLoader(df_docs, page_content_column="response_text").load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
   
    rds = Redis.from_documents(
        docs, embeddings, redis_url=settings.BROKER_URL, index_name="survey_data"
    )
    print(rds.client.dbsize()) 


if __name__ == "__main__":
    main()
