import os
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

dotenv_path = '../.env'
load_dotenv(dotenv_path)
MONGODB_ATLAS_CLUSTER_URI = os.getenv('MONGODB_URI')

if MONGODB_ATLAS_CLUSTER_URI is None:
    raise ValueError("MONGODB_URI environment variable is not set.")

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

if os.getenv('OPENAI_API_KEY') is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

class ResearchPaperDB:
    """

    ResearchPaperDB class is used to interact with a MongoDB database containing research papers.

    Parameters:
    - db_name (str): The name of the MongoDB database to connect to.
    - collection_name (str): The name of the collection in the database to use.

    Attributes:
    - client (MongoClient): The MongoDB client object used to connect to the database.
    - db (Database): The MongoDB database object.
    - collection (Collection): The MongoDB collection object.
    - embeddings (OpenAIEmbeddings): An instance of the OpenAIEmbeddings class used for embedding text.
    - text_splitter (RecursiveCharacterTextSplitter): An instance of the RecursiveCharacterTextSplitter class used for splitting text into chunks.

    Methods:
    - get_client(): Returns the MongoDB client object.
    - get_collection(): Returns the MongoDB collection object.
    - get_db(): Returns the MongoDB database object.
    - insert_paper(title, abstract, year, url, authors, external_id, open_access_pdf): Inserts a research paper document into the collection.
    - embed_query(text): Embeds a query text using the OpenAIEmbeddings class.
    - find_similar_documents(embedding, index_name, embedding_field, limit=5): Finds similar documents in the collection based on the provided embedding.

    """

    def __init__(self, db_name, collection_name):
        if not isinstance(db_name, str) or not isinstance(collection_name, str):
            raise TypeError("Both 'db_name' and 'collection_name' have to be string.")

        try:
            self.client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            self.embeddings = OpenAIEmbeddings(disallowed_special=())
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        except Exception as e:
            print("Exception while connecting to the database or initializing embeddings: ", e)

    def get_client(self):
        return self.client

    def get_collection(self):
        return self.collection

    def get_db(self):
        return self.db

    def insert_paper(self, title, abstract, year, url, authors, external_id, open_access_pdf):
        """
        Insert a paper into the database.

        :param title: The title of the paper.
        :param abstract: The abstract of the paper.
        :param year: The year of publication.
        :param url: The URL of the paper.
        :param authors: The authors of the paper.
        :param external_id: The external ID of the paper.
        :param open_access_pdf: Indicates whether the paper has open access PDF.
        :return: None
        :raises ValueError: If any of the input parameters has an invalid type.
        """
        if not all(isinstance(param, str) for param in [title, abstract, url, authors, external_id]) or not isinstance(year, int) or not isinstance(open_access_pdf, bool):
            raise ValueError("Invalid input. Check the input types.")

        if title or abstract:
            try:
                title_abstract = (title or '') + " " + (abstract or '')
                title_abstract_embedding = self.embeddings.embed_documents([title_abstract])[0]
                embedding_successful = True
            except Exception as e:
                print(f"Exception while embedding the text: {e}")
                title_abstract_embedding = None
                embedding_successful = False

            document = {
                "title": title,
                "abstract": abstract,
                "year": year,
                "url": url,
                "embedding": title_abstract_embedding,
                "embedding_successful": embedding_successful,
                "authors": authors,
                "externalIds": external_id,
                "openAccessPdf": open_access_pdf
            }

            try:
                self.collection.insert_one(document)
            except Exception as e:
                print(f"Exception while inserting the document into the database: {e}")
        else:
            print("Both title and abstract cannot be None or empty.")

    def embed_query(self, text):
        """
        Embeds the given query text using the embeddings service.

        :param text: The query text to be embedded.
        :return: The embedded representation of the query text.
        """
        return self.embeddings.embed_query(text)

    def find_similar_documents(self, embedding, index_name, embedding_field, num_candidates=50, limit=5):
        """
        :param embedding: The embedding vector of the target document.
        :type embedding: list

        :param index_name: The name of the index to search for similar documents.
        :type index_name: str

        :param embedding_field: The field that contains the vectors to compare against.
        :type embedding_field: str

        :param num_candidates: The maximum number of candidate documents to consider during the search. Default is 50.
        :type num_candidates: int

        :param limit: The maximum number of similar documents to return. Default is 5.
        :type limit: int

        :return: A list of similar documents that match the given query vector.
        :rtype: list

        """
        try:
            documents = list(self.collection.aggregate([
                {
                    "$vectorSearch": {
                        "queryVector": embedding,
                        "path": embedding_field,
                        "numCandidates": num_candidates,
                        "limit": limit,
                        "index": index_name
                    }
                }
            ]))
            return documents
        except Exception as e:
            print(f"Exception during the searching for similar documents: {e}")