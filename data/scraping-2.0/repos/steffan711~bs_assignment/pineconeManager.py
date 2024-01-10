import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from uuid import uuid4
from langchain.vectorstores import Pinecone

class PineconeManager:
    """
    A class to manage Pinecone indexing and vector storage.
    """

    def __init__(self, model_name, pine_api_key, openai_api_key,
                 environment, index_name, metric, dimension, text_field):
        """
        Initialize the PineconeManager with model, API keys, and index configurations.

        :param model_name: Name of the embedding model.
        :param pine_api_key: API key for Pinecone.
        :param openai_api_key: API key for OpenAI.
        :param environment: Environment for Pinecone.
        :param index_name: Name of the Pinecone index.
        :param metric: The metric for Pinecone index.
        :param dimension: The dimension for Pinecone index.
        :param text_field: The text field to be indexed.
        """
        # Text processing initialization
        self.tokenizer = tiktoken.get_encoding('p50k_base')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=self.tiktoken_len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embed = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=openai_api_key
        )

        # Pinecone index management initialization
        self.api_key = pine_api_key
        self.environment = environment
        self.index_name = index_name
        self.metric = metric
        self.dimension = dimension
        self.text_field = text_field
        self.initialize_pinecone()
        self.create_index()

        self.vectorstore = Pinecone(
            self.index, self.embed, text_field
        )

    def tiktoken_len(self, text):
        """
        Calculate the length of a text in tokens.

        :param text: The text to be tokenized.
        :return: Length of the tokenized text.
        """
        tokens = self.tokenizer.encode(text, disallowed_special=())
        return len(tokens)

    def fill_index(self, data_with_citations):
        """
        Fill the Pinecone index with data.

        :param data_with_citations: Data containing texts and citations.
        """
        if self.index_not_empty():
            print("Index is not empty")
            return

        batch_limit = 100 # Recommended by Pinecone documentation
        texts = []
        metadatas = []

        for record in data_with_citations:
            metadata = {'source': record['url']}
            record_texts = self.text_splitter.split_text(record['text'])
            record_metadatas = [{"text": text, **metadata} for text in record_texts]
            texts.extend(record_texts)
            metadatas.extend(record_metadatas)

            if len(texts) >= batch_limit:
                ids = [str(uuid4()) for _ in range(len(texts))]  # Generate unique IDs
                embeds = self.embed.embed_documents(texts)
                self.index.upsert(vectors=list(zip(ids, embeds, metadatas)))
                texts = []
                metadatas = []

    def initialize_pinecone(self):
        """
        Initialize Pinecone with the provided API key and environment.
        """
        pinecone.init(api_key=self.api_key, environment=self.environment)

    def create_index(self):
        """
        Create a Pinecone index if it doesn't exist.
        """
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                metric=self.metric,
                dimension=self.dimension
            )
        self.index = pinecone.Index(self.index_name)

    def delete_index(self):
        """
        Delete the Pinecone index if it exists.
        """
        if self.index_name in pinecone.list_indexes():
            pinecone.delete_index(self.index_name)

    def index_not_empty(self):
        """
        Check whether index contains any vectors.

        :return: Statistics of the index.
        """

        if self.index_name in pinecone.list_indexes():
            return self.index.describe_index_stats()['total_vector_count'] > 0
        else:
            return False

    def similarity_search(self, query, k):
        """
        Perform a similarity search in the Pinecone index.

        :param query: Query text.
        :param k: Number of similar items to return.
        :return: Results of the similarity search.
        """
        return self.vectorstore.similarity_search(query, k)
