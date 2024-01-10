"""Loader that loads data from JSON."""
import json
from pathlib import Path
from typing import List, Optional, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

import os
import keys

os.environ['OPENAI_API_KEY'] = keys.openAI_APIKEY

class JSONLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        content_key: Optional[str] = None,
        ):
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key
        self._content_key = content_key

    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""

        docs=[]
        # Load JSON file
        with open(self.file_path) as file:
            data = json.load(file)

            # Iterate through conversations
            for conversationIndex, conversation in enumerate(data):
                conversationData = data[conversation]
                
                context = ""
                for msgIndex, msg in enumerate(conversationData):
                    ID = msg['id']
                    sender = msg['sender']
                    commentQuotedID = msg['commentQuotedID']
                    message = msg['message']
                    sequenceNumber = msgIndex
                    conversationID = conversation
                    #context = ""

                    #for otherMsg in conversationData:
                    context += msg["sender"] + ": " + msg["message"] + "\n"

                #metadata = dict(sender=sender, conversationID=conversationID, context=context)
                
                docs.append(Document(page_content=context, metadata=dict()))
        return docs
    
    def loadResponses(self, jsondata) -> List[Document]:
        """Load and return documents from the JSON file."""

        docs=[]
        data = jsondata

        # Iterate through responses
        for response in data["responses"]:
            AIresponse = response["AIresponse"]
            context = response["context"]
            score = response["score"]

            metadata = dict(AIresponse=AIresponse, score=score)
            
            docs.append(Document(page_content=context, metadata=metadata))

        return docs