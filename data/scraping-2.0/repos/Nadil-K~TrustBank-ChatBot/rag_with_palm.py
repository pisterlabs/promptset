from llama_index import SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms.palm import PaLM
from llama_index import ServiceContext
from llama_index import StorageContext
import os
from llama_index.llms import OpenAI

class RAGPaLMQuery:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RAGPaLMQuery, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        # Create a folder for data if it doesn't exist
        # if not os.path.exists("data"):
        #     os.makedirs("data")

        # Load documents from the data folder
        self.documents = SimpleDirectoryReader("./competition").load_data()

        # Set up API key for PaLM
        # os.environ['GOOGLE_API_KEY'] = 'AIzaSyCU-_6aIDqcFzwwJrT8r5g585sHTZ0MZhY'
        os.environ['OPENAI_API_KEY'] = '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'
        
        self.llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")

        # Initialize PaLM and Hugging Face embedding model
        # self.llm = PaLM()
        # self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
        self.embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-large-en-v1.5')
        
        # Set up service context
        self.service_context = ServiceContext.from_defaults(llm=self.llm, embed_model=self.embed_model, chunk_size=800, chunk_overlap=20)

        # Create a VectorStoreIndex from the documents
        self.index = VectorStoreIndex.from_documents(self.documents, service_context=self.service_context)


        # Create a query engine
        self.query_engine = self.index.as_query_engine(similarity_top_k=10)

    def query_response(self, query):
        # Perform a query
        response = self.query_engine.query(query)
        return response

