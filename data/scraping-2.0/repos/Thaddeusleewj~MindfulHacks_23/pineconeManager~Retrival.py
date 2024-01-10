from langchain.vectorstores import Pinecone
import openai
import pinecone
import tqdm


class PineconeRetrival:
    """Wraps around main functionality of upsert text to pinecone index"""

    def __init__(self, PINECONE_API_KEY, PINECONE_ENVIRONMENT,INDEX_NAME):

        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        if INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(INDEX_NAME, dimension=1536)

        self.index = pinecone.Index(INDEX_NAME)
        self.embedding_model = "text-embedding-ada-002"
        docsearch = Pinecone.from_existing_index(INDEX_NAME, self.embedding_model)
        self.retriever = docsearch.as_retriever(search_type="mmr")

        index_stats_response = self.index.describe_index_stats()
        print('index_stats_response: ', index_stats_response)
        index_description = pinecone.describe_index(INDEX_NAME)
        print('index_description: ', index_description)


    def getPineconeRelevantDocuments(self,query:str):
        """
        Parameters:
            query (str): query to search for relevant documents in database
        Returns:
            relevant_documents (str): relevant documents in database based on query
        """
        relevant_documents = ""
        matched_docs = self.retriever.get_relevant_documents(query)

        for i, d in enumerate(matched_docs):
            # print(f"\n## Document {i}\n")
            # print(d.page_content)
            relevant_documents += f'\n## Document {i}\n {d.page_content}'
            
        return relevant_documents