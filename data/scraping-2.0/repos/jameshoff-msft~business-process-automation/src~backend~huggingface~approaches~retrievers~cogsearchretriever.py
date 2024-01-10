
from approaches.callback import MyCallbackHandler
from langchain.schema import Document, BaseRetriever
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.core.credentials import AzureKeyCredential
from typing import List
import json
import os

class CogSearchRetriever(BaseRetriever):
    def __init__(self, index : str, searchables, top : int, handler : MyCallbackHandler ):
       self.index = index
       self.searchables = searchables
       self.top = top
       self.handler = handler
    def get_relevant_documents(self, query: str) -> List[Document]:
        search_client = SearchClient(
            endpoint="https://"+os.environ["AZURE_SEARCH_SERVICE"]+".search.windows.net",
            index_name=self.index.get("name"),
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_APIKEY"]))
        
        semanticConfigs = self.index.get("semanticConfigurations") or []
        if len(semanticConfigs) > 0:
            semantic_name = semanticConfigs[0]["name"]
            r = search_client.search(query, 
                query_type=QueryType.SEMANTIC, 
                query_language="en-us", 
                query_speller="lexicon", 
                facets=self.index.get("facetableFields"),
                semantic_configuration_name=semantic_name, 
                top=self.top, 
                include_total_count=True,
                query_caption="extractive|highlight-false")
        else:
            r = search_client.search(query, top=self.top)
        docs = []
        doc_names = []
        for doc in r:
            
            doc["source"] = doc["filename"]
            text = self.nonewlines(self.getText(self.searchables, doc))
            doc_names.append("{} : {}".format(doc["filename"], text ))
            docs.append(Document(page_content=text, metadata={"source":doc["filename"]}))
        self.handler.on_llm_start([],doc_names)
        return docs
    
    def nonewlines(self, s: str) -> str:
        return s.replace('\n', ' ').replace('\r', ' ')
    
    def getText(self, searchables, doc):
        if searchables == None:
            return ""
        if len(searchables) == 0:
            return ""
        out = ""
        for s in searchables:
            currentData = doc
            for i in s.split('/'):
                if  isinstance(currentData.get(i), list):
                    currentData = currentData.get(i)[0]
                else:
                    currentData = currentData[i]
                if isinstance(currentData, str):
                    out = out + currentData
                
        return out
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        
    
        return self.get_relevant_documents(query)