import os, requests, tiktoken

from llama_hub.youtube_transcript import YoutubeTranscriptReader
from llama_index import download_loader, PromptTemplate, ServiceContext
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index.embeddings import OpenAIEmbedding
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from pathlib import Path
from pymongo import MongoClient
from rag_base import BaseRAG

class LlamaIndexRAG(BaseRAG):
    MONGODB_DB_NAME = "llamaindex_db"

    def load_documents(self):
        docs = []
    
        # PDF
        PDFReader = download_loader("PDFReader")
        loader = PDFReader()
        out_dir = Path("data")
    
        if not out_dir.exists():
            os.makedirs(out_dir)
    
        out_path = out_dir / "gpt-4.pdf"
    
        if not out_path.exists():
            r = requests.get(self.PDF_URL)
            with open(out_path, "wb") as f:
                f.write(r.content)

        docs.extend(loader.load_data(file = Path(out_path)))
        #print("docs = " + str(len(docs)))
    
        # Web
        SimpleWebPageReader = download_loader("SimpleWebPageReader")
        loader = SimpleWebPageReader()
        docs.extend(loader.load_data(urls = [self.WEB_URL]))
        #print("docs = " + str(len(docs)))

        # YouTube
        loader = YoutubeTranscriptReader()
        docs.extend(loader.load_data(ytlinks = [self.YOUTUBE_URL_1,
                                                self.YOUTUBE_URL_2]))
        #print("docs = " + str(len(docs)))
    
        return docs

    def get_callback_manager(self, config):
        token_counter = TokenCountingHandler(
            tokenizer = tiktoken.encoding_for_model(config["model_name"]).encode
        )

        token_counter.reset_counts()

        return CallbackManager([token_counter])

    def get_callback(self, token_counter):
        return ("Tokens Used: " +
                str(token_counter.total_llm_token_count) + "\n" +
                "Prompt Tokens: " +
                str(token_counter.prompt_llm_token_count) + "\n" +
                "Completion Tokens: " +
                str(token_counter.completion_llm_token_count))

    def get_llm(self, config):
        return OpenAI(
            model = config["model_name"], 
            temperature = config["temperature"]
        )

    def get_vector_store(self):
        return MongoDBAtlasVectorSearch(
            MongoClient(self.MONGODB_ATLAS_CLUSTER_URI),
            db_name = self.MONGODB_DB_NAME,
            collection_name = self.MONGODB_COLLECTION_NAME,
            index_name = self.MONGODB_INDEX_NAME
        )
        
    def get_service_context(self, config):
        return ServiceContext.from_defaults(
            callback_manager = self.get_callback_manager(config),
            chunk_overlap = config["chunk_overlap"],
            chunk_size = config["chunk_size"],
            embed_model = OpenAIEmbedding(), # embed
            llm = self.get_llm(config)
        )

    def get_storage_context(self):
        return StorageContext.from_defaults(
            vector_store = self.get_vector_store()
        )
        
    def store_documents(self, config, docs):
        storage_context = StorageContext.from_defaults(
            vector_store = self.get_vector_store()
        )
    
        VectorStoreIndex.from_documents(
            docs,
            service_context = self.get_service_context(config),
            storage_context = self.get_storage_context()
        )
  
    def ingestion(self, config):
        docs = self.load_documents()
    
        self.store_documents(config, docs)
       
    def retrieval(self, config, prompt):
        index = VectorStoreIndex.from_vector_store(
            vector_store = self.get_vector_store()
        )

        service_context = self.get_service_context(config)
        
        query_engine = index.as_query_engine(
            service_context = service_context,
            similarity_top_k = config["k"],
            text_qa_template = PromptTemplate(os.environ["LLAMAINDEX_TEMPLATE"])
        )

        completion = query_engine.query(prompt)
        callback = self.get_callback(
            service_context.callback_manager.handlers[0])
        
        return completion, callback
