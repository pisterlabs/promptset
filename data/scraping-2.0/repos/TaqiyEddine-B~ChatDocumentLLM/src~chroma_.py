""" """
import os
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from dotenv import load_dotenv
load_dotenv()


class ChromaPy:

    def __init__(self, openai_key):
        self.db = None
        self.raw_text = None
        # check if the env variable 'OPENAI_API_KEY' is set
        if not os.getenv('OPENAI_API_KEY'):
            # if not, set it to the value of the openai_key parameter
            os.environ['OPENAI_API_KEY'] = openai_key

    def prepare(self, txt_file):
        # Load the document, split it into chunks, embed each chunk and load it into the vector store.
        raw_documents = TextLoader(txt_file).load()
        self.raw_text = raw_documents[0].page_content
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)
        db = Chroma.from_documents(documents, OpenAIEmbeddings())
        self.db = db

    def chat_function(self, query):
        docs = self.db.similarity_search(query, k=1)
        # print(docs[0].page_content)
        result = {'answer': docs[0].page_content}
        return result
