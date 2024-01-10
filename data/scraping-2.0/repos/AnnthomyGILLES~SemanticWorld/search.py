from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import ElasticVectorSearch

from config import openai_api_key

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

db = ElasticVectorSearch(
    elasticsearch_url="http://localhost:9200",
    index_name="elastic-index",
    embedding=embedding,
)
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=db.as_retriever(),
)


def ask(query: str):
    response = qa.run(query)
    return response


if __name__ == "__main__":
    response = ask("What did you learn from your father")
    print(response)
