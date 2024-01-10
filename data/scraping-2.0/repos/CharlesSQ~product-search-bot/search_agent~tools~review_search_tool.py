import pinecone
import os

from .custom_self_query_retriever.base import CustomSelfQueryRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool

from dotenv import load_dotenv

load_dotenv()

# Access environment variables
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENVIROMENT = os.environ['PINECONE_ENVIROMENT']
PINECONE_INDEX_NAME = os.environ['PINECONE_INDEX_NAME']


metadata_field_info = {}

metadata_field_info["prod_id"] = {
    "description": "The product id",
    "type": "string"
}

# metadata_field_info["genre"] = {
#     "description": "The genre of the movie",
#     "type": "string or list[string]"
# }

# metadata_field_info["year"] = {
#     "description": "The year the movie was released",
#     "type": "integer"
# }

# metadata_field_info["director"] = {
#     "description": "The name of the movie director",
#     "type": "string"
# }

# metadata_field_info["rating"] = {
#     "description": "A 1-10 rating for the movie",
#     "type": "float"
# }


class ReviewSearchTool():
    def __init__(self):

        llm_chat = ChatOpenAI(
            temperature=0.9, model='gpt-3.5-turbo-0613', client='')

        embeddings = OpenAIEmbeddings(client='')

        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIROMENT
        )

        vectorstore = Pinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME, embedding=embeddings, text_key='title', namespace='reviews')

        document_content_description = "Reviews data"

        retry_message = "No results found, ask user to be more specific. Never mention the product id or a reference to it."

        self.retriever = CustomSelfQueryRetriever.from_llm(
            llm_chat, vectorstore, document_content_description, metadata_field_info, max_retry=3, retry_message=retry_message)

    def retriever_tool(self, input):
        print('input:', input)
        k = 5

        if isinstance(input, list):
            product_ids = ''
            for prod_id in input:
                product_ids += prod_id + ', '

            query_input = f'Reviews of products with ids {product_ids}, return {k} reviews. Use the eq comparator.'

        elif isinstance(input, str):
            query_input = f'Reviews of product with id {input}, return {k} reviews. Use the eq comparator.'

        return self.retriever.get_relevant_documents(query=query_input)

    def aretriever_tool(self, input: str):
        k = 5

        if isinstance(input, list):
            product_ids = ''
            for prod_id in input:
                product_ids += prod_id + ', '

            query_input = f'Reviews of products with ids {product_ids}, return {k} reviews. Use the eq comparator.'

        elif isinstance(input, str):
            query_input = f'Reviews of product with id {input}, return {k} reviews. Use the eq comparator.'

        return self.retriever.aget_relevant_documents(query=query_input)

    def get_tool(self):
        return Tool(
            name="search_reviews",
            description="Searches and returns product reviews from the database. Input: {'prod_ids': An array of product ids}",
            func=self.retriever_tool,
            coroutine=self.aretriever_tool
        )


# def main():
#     query = 'reviews de productos con ids 63b260653d66c49aca71d738 y 64439cf4dad6ccfb902d7327, retorna 5 reviews'
#     input = "I want to watch a movie rated higher than 8.5"
#     # user_prompt = input("Usuario: ")
#     agent = ReviewSearchTool().retriever_tool
#     response = agent(query)
#     print(f'\n\nRetriever: ${response}')


# if __name__ == '__main__':
#     main()
