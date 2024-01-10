import json
from typing import List, Sequence, Any
from abc import ABC, abstractmethod

from langchain.docstore.document import Document
from langchain.schema import BaseDocumentTransformer
class JSONSplitter(BaseDocumentTransformer, ABC):

    def split_document(self, document: Document) -> List[Document]:
        # Attempt to parse the JSON content of the document
        try:
            content = json.loads(document.page_content)
        except json.JSONDecodeError:
            # Handle invalid JSON
            return []

        # Check if content is a list of objects
        if not isinstance(content, list):
            content = [content]

        # Create a new Document for each object in the list
        documents = []
        for item in content:
            if isinstance(item, dict):
                # Convert the dict to a JSON string
                new_page_content = json.dumps(item)
                new_document = Document(page_content=new_page_content, metadata={}, type="Document")
                documents.append(new_document)

        return documents

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        transformed_documents = []
        for document in documents:
            split_docs = self.split_document(document)
            transformed_documents.extend(split_docs)
        return transformed_documents

    async def atransform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        # Implement asynchronous handling if needed
        return await super().atransform_documents(documents, **kwargs)
