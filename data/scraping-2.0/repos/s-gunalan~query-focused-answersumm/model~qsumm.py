import os
import textwrap
from typing import List
import time
import langchain
from langchain.chains import RetrievalQA
from langchain.document_loaders import GCSDirectoryLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain.embeddings import VertexAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

import requests
from google.cloud import aiplatform
import vertexai

os.environ['OPENAI_API_KEY'] = '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'

class QueryAnsweringService:
    def __init__(self):
        # Initialize necessary components and variables here
        self.llm = None
        self.embeddings = None
        self.me = None
        self.qa = None

    def initialize_service(self,data):
        # Initialize the service and required dependencies
        self.initialize_vertex_ai_sdk()
        self.initialize_chroma(data)
        #self.initialize_langchain()
        #self.initialize_matching_engine()

    def initialize_vertex_ai_sdk(self):
        self.PROJECT_ID = "alkali-gworks"  # @param {type:"string"}
        self.REGION = "us-central1"  # @param {type:"string"}
        #vertexai.init(project=self.PROJECT_ID, location=self.REGION)
        self.llm = VertexAI(model_name="text-bison@001")
        

    def initialize_chroma(self,data):
        self.vertex_embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@001")
        from langchain.embeddings.openai import OpenAIEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, add_start_index=True)
        documents = text_splitter.split_documents(data[:100])
        self.vectordb = Chroma.from_documents(documents, self.vertex_embeddings, persist_directory="./chroma_dbj1").as_retriever()

    def initialize_langchain(self):
        # Initialize langChain components
        self.llm = VertexAI(
            model_name="text-bison@001",
            max_output_tokens=1024,
            temperature=0.2,
            top_p=0.8,
            top_k=40,
            verbose=True,
        )

        EMBEDDING_QPM = 100
        EMBEDDING_NUM_BATCH = 5
        self.embeddings = CustomVertexAIEmbeddings(
            requests_per_minute=EMBEDDING_QPM,
            num_instances_per_batch=EMBEDDING_NUM_BATCH,
        )
    def initialize_matching_engine(self):
        ME_REGION = "us-central1"
        ME_INDEX_NAME = f"{self.PROJECT_ID}-gr-index-d4"  # @param {type:"string"}
        ME_EMBEDDING_DIR = f"{self.PROJECT_ID}-gr-bucket-d3"  # @param {type:"string"}
        ME_DIMENSIONS = 768  # when using Vertex PaLM Embedding
        ME_EMBEDDING_DIR = 'gs://gworks-vector-index/langchain/embeddings'

        from utils.matching_engine_utils import MatchingEngineUtils
        from utils.matching_engine import MatchingEngine

        self.mengine = MatchingEngineUtils(self.PROJECT_ID, ME_REGION, ME_INDEX_NAME)
        GCS_BUCKET_DOCS = "gworks-vector-index"
        folder_prefix = "langchain/documents"
        ME_INDEX_ID, ME_INDEX_ENDPOINT_ID = self.mengine.get_index_and_endpoint()
        print(f"ME_INDEX_ID={ME_INDEX_ID}")
        print(f"ME_INDEX_ENDPOINT_ID={ME_INDEX_ENDPOINT_ID}")

        # Initialize Matching Engine
        self.me = MatchingEngine.from_components(
            project_id=self.PROJECT_ID,
            region=ME_REGION,
            gcs_bucket_name=f'gs://{ME_EMBEDDING_DIR.split("/")[2]}',
            embedding=self.embeddings,
            index_id=ME_INDEX_ID,
            endpoint_id=ME_INDEX_ENDPOINT_ID,
        )

    def rate_limit(self, max_per_minute):
        # Rate limiting implementation
        period = 60 / max_per_minute
        print("Waiting")
        while True:
            before = time.time()
            yield
            after = time.time()
            elapsed = after - before
            sleep_time = max(0, period - elapsed)
            if sleep_time > 0:
                print(".", end="")
                time.sleep(sleep_time)

    def set_environment_variable(self, variable_name, variable_value):
        # Set environment variables
        os.environ[variable_name] = variable_value

    def formatter(self, result):
        # Format and display the result
        print(f"Query: {result['query']}")
        print("." * 80)
        if "source_documents" in result.keys():
            for idx, ref in enumerate(result["source_documents"]):
                print("-" * 80)
                print(f"REFERENCE #{idx}")
                print("-" * 80)
                if "score" in ref.metadata:
                    print(f"Matching Score: {ref.metadata['score']}")
                if "source" in ref.metadata:
                    print(f"Document Source: {ref.metadata['source']}")
                if "document_name" in ref.metadata:
                    print(f"Document Name: {ref.metadata['document_name']}")
                print("." * 80)
                print(f"Content: \n{self.wrap(ref.page_content)}")
        print("." * 80)
        print(f"Response: {self.wrap(result['result'])}")
        print("." * 80)

    def wrap(self, s):
        # Wrap long text
        return "\n".join(textwrap.wrap(s, width=120, break_long_words=False))
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    def ask(self, query):
        # Ask a question and get the answer
        NUMBER_OF_RESULTS = 10
        SEARCH_DISTANCE_THRESHOLD = 0.6

        prompt = """SYSTEM: You are an intelligent assistant helping the users with their questions.

       
        Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.

        Do not try to make up an answer:
         - If the answer to the question cannot be determined from the context alone, say "I cannot determine the answer to that."
         - If the context is empty, just say "I do not know the answer to that."
        Do not reduce the content from the context mostly or do not try to summarise the answer

        =============
        {context}
        =============

        Question: {question}
        Helpful Answer:"""

        retriever = self.vectordb
        rag_chain = (
            {"context": retriever , "question": query}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        inputs = {"question": query}
        result = rag_chain.invoke(inputs)
        self.formatter(result)       
        bot_response = {
            'result': result['result'],
        }
        return bot_response
    def load_db(self, file,source_url):
        # load documents
        # Load text data from a file using TextLoader
        loader = TextLoader(file)
        documents = loader.load()

        # Add document name and source to the metadata
        for document in documents:
            doc_md = document.metadata
            document_name = doc_md["source"]
            # derive doc source from Document loader
            doc_source_suffix = "/".join(doc_md["source"].split("/")[4:-1])
            if source_url:
                source = source_url
            else:
                source = f"{doc_source_suffix}"
            document.metadata = {"source": source, "document_name": document_name}


        # split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        doc_splits = text_splitter.split_documents(documents)
        texts = [doc.page_content for doc in doc_splits]
        metadatas = [
            [
                {"namespace": "source", "allow_list": [doc.metadata["source"]]},
                {"namespace": "document_name", "allow_list": [doc.metadata["document_name"]]},
            ]
            for doc in doc_splits
        ]
        print(texts)
        print(metadatas)
        me = self.me
        me.add_texts(texts=texts, metadatas=metadatas)
        print("Embeddings Added")
