from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import os
import pinecone

from ultimate_frisbee_rules_rag import logger
from ultimate_frisbee_rules_rag.config.configuration import ConfigurationManager


class InitialiseVectorStorePipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_initialise_vectorstore_config()

    def main(self):
        config = ConfigurationManager().get_initialise_vectorstore_config()
        pipeline = InitialiseVectorStorePipeline(config)
        pipeline.initialise()


if __name__ == "__main__":
    try:
        logger.info("Initialising vectorstore")
        InitialiseVectorStorePipeline().main()
        logger.info("Initialised vectorstore")
    except Exception as e:
        logger.exception(f"Error while initialising vectorstore: {e}")
        raise e
