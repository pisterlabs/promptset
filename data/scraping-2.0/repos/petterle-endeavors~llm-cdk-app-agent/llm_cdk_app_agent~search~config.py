import os
import pinecone
from aws_lambda_powertools.utilities import parameters
from indexer.fetch_cdk import read_init, split_docs
from indexer.fastapi_data import split_fastapi_docs
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["OPENAI_API_KEY"] = str(parameters.get_secret("openai"))
pinecone_secret = str(parameters.get_secret("pinecone"))


class PineconeManager:
    "Managing pinecone configurations!"

    def __init__(self, api_key=pinecone_secret, env_key="us-east-1-aws", index_name="llm-cdk-agent"):
        self.api_key = api_key
        self.env_key = env_key
        self.index_name = index_name
        pinecone.init(api_key=self.api_key, environment=self.env_key)

    def list_indexes(self):
        "Getting all indexes and returns as a list."
        return pinecone.list_indexes()

    def create_or_get_index(self, dimension=1536, metric="cosine"):
        "Creation of our vector database index!"
        if self.index_name not in self.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
            )
            print(f"created a new index {self.index_name}")
        else:
            print(f"{self.index_name} index existed. Skip creating.")
            index = pinecone.Index(self.index_name)
            index_stats_response = index.describe_index_stats()
            vector_count = index_stats_response.total_vector_count
            print(f"total_vector_count is {vector_count}")

        return pinecone.Index(self.index_name)


if __name__ == "__main__":
    pico = PineconeManager()
    index = pico.create_or_get_index()
    # test, _ = read_init()
    # embeddings = OpenAIEmbeddings()
    # vectorstore = Pinecone(index, embeddings.embed_query, "text")
    # vectorstore.add_documents(split_fastapi_docs(), namespace="fastapi-docs")
    # #pinecone.delete_index("llm-cdk-agent")
    print("everything uploaded successfuly!")
