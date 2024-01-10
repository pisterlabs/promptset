from langchain.document_loaders import BSHTMLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch
from config import Paths, openai_api_key, elastic_search_url

def main():
    loader = BSHTMLLoader(str(Paths.book))
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    documents = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = ElasticVectorSearch.from_documents(
        documents,
        embeddings,
        elasticsearch_url=f"{elastic_search_url}",
        index_name="elastic-index",
    )
    print(db.client.info())


if __name__ == "__main__":
    main()
