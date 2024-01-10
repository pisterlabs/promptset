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

metadata_field_info["_id"] = {
    "description": "The product id",
    "type": "string"
}


class ProductInfoTool():
    def __init__(self):

        llm_chat = ChatOpenAI(
            temperature=0.9, model='gpt-3.5-turbo-0613', client='')

        embeddings = OpenAIEmbeddings(client='')

        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIROMENT
        )

        vectorstore = Pinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME, embedding=embeddings, text_key='title', namespace='products')

        document_content_description = "Products data"

        retry_message = "No results found, ask user to be more specific and don't mention de product id"

        self.retriever = CustomSelfQueryRetriever.from_llm(
            llm_chat, vectorstore, document_content_description, metadata_field_info, max_retry=3, retry_message=retry_message)

    def retriever_tool(self, input):
        print('input:', input)
        k = 5

        if isinstance(input, list):
            product_ids = ''
            for prod_id in input:
                product_ids += prod_id + ', '

            query_input = f'Give me all data of product with ids {product_ids}, return {k} reviews. Use the eq comparator.'

        elif isinstance(input, str):
            query_input = f'GIve me all data of product with id {input}, return {k} reviews.'

        return self.retriever.get_relevant_documents(query=query_input)

    def aretriever_tool(self, input: str):
        k = 5

        if isinstance(input, list):
            product_ids = ''
            for prod_id in input:
                product_ids += prod_id + ', '

            query_input = f'Give me all data of product with ids {product_ids}, return {k} reviews. Use the eq comparator.'

        elif isinstance(input, str):
            query_input = f'GIve me all data of product with id {input}, return {k} reviews.'

        return self.retriever.aget_relevant_documents(query=query_input)

    def get_tool(self):
        return Tool(
            name="get_product_info",
            description="Useful for when you need to search for products based on product id. Input: {'prod_ids': An array of product ids}",
            func=self.retriever_tool,
            coroutine=self.aretriever_tool
        )


# def main():
#     query = 'dame los ingredientes de productos con ids 63b260653d66c49aca71d738 y 64439cf4dad6ccfb902d7327, retorna 5 resultados'
#     input = 'Give me informaction of products with ids 63ab662808edd368196f33a2, 63b260663d66c49aca71dc2b, return 5 reviews.'
#     # user_prompt = input("Usuario: ")
#     agent = ProductInfoTool().retriever.get_relevant_documents
#     response = agent(input)
#     print(f'\n\nRetriever: ${response}')


# if __name__ == '__main__':
#     main()
