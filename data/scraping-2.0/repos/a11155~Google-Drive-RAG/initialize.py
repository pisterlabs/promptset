
from dotenv import load_dotenv
from llama_index import VectorStoreIndex
from logging_spinner import SpinnerHandler
from llama_index.llms import OpenAI
from llama_index.node_parser import SentenceWindowNodeParser, SimpleNodeParser
from llama_index import ServiceContext, set_global_service_context
from llama_index.embeddings import HuggingFaceEmbedding, OpenAIEmbedding
from llama_index.text_splitter import SentenceSplitter
import logging
import os
from googleDriveReader import GoogleDriveReader

def readGoogleDrive(folder_id):
    load_dotenv()

    def convert_pdf_to_html():
        pass
    
    def get_nodes(documents, node_parser):
        nodes = node_parser.get_nodes_from_documents(documents)
        return nodes


    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3, # one large document vs many little documents
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )



    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(SpinnerHandler())


    folder_id = folder_id #os.getenv('FOLDER_ID')

    loader = GoogleDriveReader()

    logger.info('Loading data...', extra={'user_waiting': True})
    documents = loader.load_data(folder_id=folder_id)
    logger.info('Finished loading!', extra={'user_waiting': False})

    nodes = get_nodes(documents, node_parser)

    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    ctx = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0),
        embed_model=OpenAIEmbedding(embed_batch_size=50),
        text_splitter=text_splitter)


    index = VectorStoreIndex(nodes, service_context=ctx)
    index.storage_context.persist()

    return index