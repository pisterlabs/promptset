from pathlib import Path
from llama_index import Document, SimpleDirectoryReader, download_loader
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import GPTVectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores import PineconeVectorStore
import pinecone
import os
from llama_index.node_parser import SimpleNodeParser
import openai
from dotenv import load_dotenv
from os import getenv




def query_research(message):
    load_dotenv()

    #openai.api_key_path = getenv('OPENAI_API_KEY')


    #constructor
    #  def __init__(
    #         self,
    #         api_key,
    #         api_username,
    #         openai_api_key=None,
    #         base_url='https://forum.subspace.network',
    #         verbose=True,
    #     ):

    #load PDF
    # PDFReader = download_loader("PDFReader")
    # loader = PDFReader()
    # docs = loader.load_data(file=Path('../../data/whitepaper.pdf'))

    docs = SimpleDirectoryReader('/Users/ryanyeung/Code/Crypto/SupportGPT/supportgpt/sources/data').load_data()
    #parse PDF
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(docs)

    # initialize connection to pinecone
    # pinecone.init(
    #     getenv('PINECONE_API_KEY'),
    #     getenv('PINECONE_ENVIRONMENT'),
    # )

    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],
                environment=os.environ['PINECONE_ENVIRONMENT']
    )

    # create the index if it does not exist already
    index_name = 'research-test'
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine'
        )

    # connect to the index
    pinecone_index = pinecone.Index(index_name)
        

    # we can select a namespace (acts as a partition in an index)
    namespace = '' # default namespace
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)


    # setup our storage (vector db)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    # setup the index/query process, ie the embedding model (and completion if used)
    embed_model = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)
    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    index = GPTVectorStoreIndex.from_documents(
        docs, storage_context=storage_context,
        service_context=service_context
    )

    # retriever = index.as_retriever(retriever_mode='default')
    # query_engine = RetrieverQueryEngine(retriever)
    # #query_engine = RetrieverQueryEngine.from_args(retriever, response_mode='default')

    query_engine = index.as_query_engine()
    res = query_engine.query(message)

    return str(res)


    # print(str(res))
    # print(res.get_formatted_sources())

    #pinecone.delete_index(index_name)