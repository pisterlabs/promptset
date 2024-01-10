from pydantic import BaseModel
import openai
import chromadb
import os

from typing import Optional
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.document_loaders import PyPDFLoader

from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter,DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever

from django.conf import settings


class getTableOfContents:
    def __init__(self, collection_name):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        ##for a collection which was created before
        self.openai_ef = OpenAIEmbeddings()
        self.persist_dir = "/home/kazuki/study/Study/5/Chroma"
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        #self.client = chromadb.PersistentClient(path=settings.PERSIST_DIR)
        self.vectorstore = Chroma(collection_name=collection_name, client=self.client, embedding_function=self.openai_ef)
        self.retriever = self.vectorstore.as_retriever()
        ##compressing
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=".")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.openai_ef)
        relevant_filter = EmbeddingsFilter(embeddings=self.openai_ef, similarity_threshold=0.76)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )

        self.compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=self.retriever)

    def contains_or_not(self, defined_variable, query):
        #prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a world class algorithm for extracting information in structured formats."),
                ("human", "Use the given format to extract information from the following input: {input}"),
                ("human", "Tip: Make sure to answer in the correct format"),
            ]
        )       

        compressed_docs = self.compression_retriever.get_relevant_documents(query)
        chain_input = f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(compressed_docs)])

        #chain
        chain = create_structured_output_chain(defined_variable, self.llm, prompt, verbose=True)
        print("beforechain\n")
        #print(chain_input)
        table_of_contents = chain.run(chain_input)

        print(table_of_contents)
        return table_of_contents
    
    #queryで対応できない形式での返答が必要な場合は項目ごとにメソッドを作る必要あり
    def create_content(self, query):
        pdf_qa = ConversationalRetrievalChain.from_llm(self.llm, self.compression_retriever, return_source_documents=True)
        
        chat_history = []

        result = pdf_qa({"question": query, "chat_history": chat_history})
        
        return result["answer"] 
    
    