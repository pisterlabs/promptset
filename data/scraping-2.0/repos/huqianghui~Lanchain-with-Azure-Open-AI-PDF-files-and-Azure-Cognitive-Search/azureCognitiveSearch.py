"""Retriever wrapper for Azure Cognitive Search."""
from __future__ import annotations

from typing import  List

import os

from langchain.schema import  Document
from langchain.retrievers import AzureCognitiveSearchRetriever
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient 
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import Vector
from langchain.embeddings import OpenAIEmbeddings


# Configure environment variables  
service_endpoint = os.environ.get("AZURE_COGNITIVE_SEARCH_SERVICE_ENDPOINT")
index_name = os.environ.get("AZURE_COGNITIVE_SEARCH_INDEX_NAME")
key = os.environ.get("AZURE_COGNITIVE_SEARCH_API_KEY")

credential = AzureKeyCredential(key)

index_client = SearchIndexClient(
    endpoint=service_endpoint, credential=credential)

embeddings = OpenAIEmbeddings(model=os.environ.get("AZURE_OPENAI_EMBEDDINGS_MODEL"), chunk_size=1, openai_api_version=os.environ.get("AZURE_OPENAI_EMBEDDINGS_API_VERSION"))

class AzureCognitiveSearchDocument():
    def __init__(self, title, score,page_content,url):
        self.title = title
        self.score = score
        self.page_content= page_content
        self.source = url
    
    @property
    def metadata(self):
        metadata = {
            "title": self.title,
            "score": self.score,
            "source": self.source
        }
        return metadata

    @metadata.setter
    def metadata(self, value):
        self.metadata = value

def generate_result_List(results):
    resultList = []
    for result in results:
        temp = AzureCognitiveSearchDocument(result["title"], result["@search.score"], result["content"], result["url"])
        resultList.append(temp)
    return resultList

class AzureCognitiveSearch6Retriever(AzureCognitiveSearchRetriever):
    """Wrapper around Azure Cognitive Search."""

    def _search(self, query: str) -> List[dict]:
        search_client = SearchClient(service_endpoint, index_name, credential=credential)  
        search_results = search_client.search(  
            search_text=query,  
            select=["title", "content", "url"],
            top=6
        )    
        docList= generate_result_List(search_results)
        if(len(docList) > 0 ):
            return docList
        else:
            raise Exception("No Document Found! Please check your query or the index.")

    async def _asearch(self, query: str) -> List[dict]:
        
        return self._search(query)

    def get_relevant_documents(self, query: str) -> List[Document]:
        search_results = self._search(query)

        return search_results

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        search_results = await self._asearch(query)

        return search_results

class AzureCognitiveSearchVectorRetriever(AzureCognitiveSearchRetriever):
    """Wrapper around Azure Cognitive Search."""

    def _search(self, query: str) -> List[dict]:
        search_client = SearchClient(service_endpoint, index_name, credential=credential)  
        vector = Vector(value=embeddings.embed_query(query), k=6, fields="contentVector")
        search_results = search_client.search(  
            search_text=None,  
            vectors=[vector],
            select=["title", "content", "url"],
            top=6
        )    
        docList= generate_result_List(search_results)
        if(len(docList) > 0 ):
            return docList
        else:
            raise Exception("No Document Found! Please check your query or the index.")

    async def _asearch(self, query: str) -> List[dict]:
        
        return self._search(query)

    def get_relevant_documents(self, query: str) -> List[Document]:
        search_results = self._search(query)

        return search_results

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        search_results = await self._asearch(query)

        return search_results

class AzureCognitiveSearchHybirdRetriever(AzureCognitiveSearchRetriever):
    """Wrapper around Azure Cognitive Search."""

    def _search(self, query: str) -> List[dict]:
        search_client = SearchClient(service_endpoint, index_name, credential=credential)  
        vector = Vector(value=embeddings.embed_query(query), k=3, fields="contentVector")
        search_results = search_client.search(  
                    search_text=query,  
                    vectors=[vector],
                    select=["title", "content", "url"],
                    top=3)  
   
        docList= generate_result_List(search_results)
        if(len(docList) > 0 ):
            return docList
        else:
            raise Exception("No Document Found! Please check your query or the index.")

    async def _asearch(self, query: str) -> List[dict]:
        
        return self._search(query)

    def get_relevant_documents(self, query: str) -> List[Document]:
        search_results = self._search(query)

        return search_results

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        search_results = await self._asearch(query)

        return search_results
    
class AzureCognitiveSearchSenamicHybirdRetriever(AzureCognitiveSearchRetriever):
    """Wrapper around Azure Cognitive Search."""

    def _search(self, query: str) -> List[dict]:
        search_client = SearchClient(service_endpoint, index_name, credential=credential)  
        vector = Vector(value=embeddings.embed_query(query), k=3, fields="contentVector")
        search_results = search_client.search(
                search_text=query,
                vectors=[vector],
                select=["title", "content", "url"],
                query_type="semantic", query_language="zh-CN", semantic_configuration_name='default', query_caption="extractive", query_answer="extractive",
                top=3
            )
   
        docList= generate_result_List(search_results)
        if(len(docList) > 0 ):
            return docList
        else:
            raise Exception("No Document Found! Please check your query or the index.")

    async def _asearch(self, query: str) -> List[dict]:
        
        return self._search(query)

    def get_relevant_documents(self, query: str) -> List[Document]:
        search_results = self._search(query)

        return search_results

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        search_results = await self._asearch(query)

        return search_results
