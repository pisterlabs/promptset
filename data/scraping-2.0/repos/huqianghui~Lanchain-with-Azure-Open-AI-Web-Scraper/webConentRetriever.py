"""Retriever wrapper for Azure Cognitive Search."""
from __future__ import annotations

from typing import  List

import os

from langchain.schema import  Document
from langchain.schema import BaseRetriever

split_size=int(os.environ.get("SPLIT_SIZE",20000))

class WebConente2LocalFileSplitDocument():
    def __init__(self, title, score,page_content,url,start_pos,end_pos):
        self.title = title
        self.score = score
        self.page_content= page_content
        self.source = url
        self.start_pos=start_pos
        self.end_pos=end_pos
    
    @property
    def metadata(self):
        metadata = {
            "title": self.title,
            "score": self.score,
            "source": self.source,
            'split_size': split_size,
            'start_position': self.start_pos,
            'end_position':self.end_pos,
        }
        return metadata

    @metadata.setter
    def metadata(self, value):
        self.metadata = value

def convert_result_List(results):
    resultList = []
    for result in results:
        temp = WebConente2LocalFileSplitDocument(result["metadata"]["file_path"],5,result["page_content"], result["metadata"]["file_path"], result["metadata"]["start_position"], result["metadata"]["end_position"])
        resultList.append(temp)
    return resultList

class WebContent2LocalFileSplitRetriever(BaseRetriever):

    """Wrapper around Local file retriever."""

    def splitFilesToDocument(self,filePath:str)-> List[dict]:
        documents = []
        with open(filePath, 'r', encoding='utf-8') as file:
            text = file.read()
            page_contents = [text[i:i+ split_size] for i in range(0, len(text), split_size)]
            
            for i, page_content in enumerate(page_contents):
                start_pos = i * split_size
                end_pos = min((i + 1) * split_size, len(text))
                metadata = {
                    'split_size': split_size,
                    'start_position': start_pos,
                    'end_position': end_pos,
                    'file_path': filePath
                }
                document={
                    "page_content": page_content,
                    "metadata": metadata
                }
                documents.append(document)

        docList = convert_result_List(documents)        
        if(len(docList) > 0 ):
            return docList
        else:
            raise Exception("No Document Found! Please check the file path or the  file.")
    
    def splitFilesListToDocument(self, filePathList: []) -> List[Document]:
        all_documents = []
        for filePath in filePathList:
            documents = self.splitFilesToDocument(filePath)
            all_documents.extend(documents)

        return all_documents
    
    def _search(self, query: str) -> List[dict]:

        if(len(os.environ.get("DOC_FILE_PATH"))>0):
            filePath=os.environ.get("DOC_FILE_PATH")
            return self.splitFilesToDocument(filePath)
        elif (len(os.environ.get("DOC_FILE_FOLDER",""))>0):
            fileFolder=os.environ.get("DOC_FILE_FOLDER","")
            filePathList = [os.path.join(fileFolder, file) for file in os.listdir(fileFolder) if not file.startswith('.') and os.path.isfile(os.path.join(fileFolder, file))]
            return self.splitFilesListToDocument(filePathList)
        else:
            raise ValueError("Either 'filePath' or 'filePathList' must be configured.")

    async def _asearch(self, query: str) -> List[dict]:
        
        return self._search(query)

    def get_relevant_documents(self, query: str) -> List[Document]:
        search_results = self._search(query)

        return search_results

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        search_results = await self._asearch(query)

        return search_results
