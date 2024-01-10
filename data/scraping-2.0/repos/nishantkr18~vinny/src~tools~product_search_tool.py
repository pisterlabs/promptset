# from typing import List
# from langchain.document_loaders import DirectoryLoader, UnstructuredPDFLoader, PyPDFLoader
# from langchain.docstore.document import Document
# from typing import List
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.chat_models import ChatOpenAI
# from langchain.vectorstores import Chroma
# from langchain.docstore.document import Document
import requests
from langchain.tools import BaseTool
# from chromadb.config import Settings
# import chromadb
# import textwrap
# from langchain.embeddings import OpenAIEmbeddings
# from langchain import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.callbacks import get_openai_callback
# from openai.error import AuthenticationError
# import logging
from pydantic import BaseModel, Field
from langchain.tools import format_tool_to_openai_function
from typing import Type
# import pinecone
# from langchain.vectorstores import Pinecone
# import os
import pandas as pd
import json


def _parse_product(products):
    products = [{
                "title": product["title"],
                "description": product["description"],
                "price": f'{product["priceRange"]["minVariantPrice"]["amount"]} {product["priceRange"]["minVariantPrice"]["currencyCode"]}',
                "onlineStoreUrl": product["onlineStoreUrl"],
                "imageUrl": product["featuredImage"]["src"],
                } for product in products]
    return products

def get_filtered_products(products: list):
    # remove keys "onlineStoreUrl" and "imageUrl" from products
    for product in products:
        del product["onlineStoreUrl"]
        del product["imageUrl"]
    return json.dumps(products)


class ProductSearchToolCheck(BaseModel):
    query: str = Field(...,
                       description='query string used to seach for products on website')
    pass


class ProductSearchTool(BaseTool):
    name = 'product_search_tool'
    description = 'Uses a query to search for products in the website'

    def _run(self, query: str) -> str:

        endpoint = "https://centralwine.myshopify.com/api/2023-07/graphql.json"
        headers = {
            "content-type": "application/json",
            "X-Shopify-Storefront-Access-Token": "6629c2aad236f5de1744a7e120c9c8bf",
        }
        graphqlQuery = {
            "query": '''
                {
                search(query: "%s", first: 50, unavailableProducts: LAST) {
                    totalCount
                    nodes {
                    ... on Product {
                        description
                        onlineStoreUrl
                        title
                        priceRange {
                        minVariantPrice {
                            amount
                            currencyCode
                        }
                        }
                        featuredImage {
                        src
                        }
                        availableForSale
                    }
                    }
                }
                }
                ''' % query,
        }

        try:
            response = requests.post(
                endpoint, headers=headers, json=graphqlQuery)
            print("fetched data!")
            products = response.json()["data"]["search"]["nodes"]
            # Select only those products that are available
            products = [product for product in products if product.get(
                "availableForSale", False)]

            # Select first 3 products
            products = products[:3]

            # Extract details from query
            products = _parse_product(products)
            return products
        except Exception as error:
            print("Request failed...")
            print(error)
            return "Request failed..."

    def _arun(self, query: str) -> str:
        raise NotImplementedError(
            'This tool does not support asynchronous execution')

    args_schema: Type[BaseModel] = ProductSearchToolCheck


if __name__ == '__main__':
    tool = ProductSearchTool()
    result = tool._run('champagne')
    print(result)
