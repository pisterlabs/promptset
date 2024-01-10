# import pinecone
# import os
# from dotenv import load_dotenv
# from utils.openai_util import ChatOpenAI
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone
#
# index_name = "bot"
# load_dotenv()
#
# class PineconeManager:
#     def __init__(self):
#         pass
#
#     def getPineConeInstance(self):
#         pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment='asia-southeast1-gcp-free')
#         if index_name in pinecone.list_indexes():
#             pinecone_index = pinecone.Index(index_name)
#         else:
#             pinecone_index = pinecone.create_index(index_name, 1536)
#         embeddings = OpenAIEmbeddings()
#         pinecone_document = Pinecone(pinecone_index, embeddings.embed_query, text_key="text")
#
#         # Create a VectorStoreRetrieverMemory instance
#         # vector_store_retriever = VectorStoreRetrieverMemory(pinecone_document, similarity_threshold=0.8)
#
#         return pinecone_document
