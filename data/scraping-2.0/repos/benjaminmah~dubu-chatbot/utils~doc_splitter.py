from typing import List, Optional
from langchain.docstore.document import Document
import copy

class DocSplitter:
    def __init__(self, tokenizer, chunk_size=1000, chunk_overlap=200):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._tokenizer = tokenizer
    
    def split_text(self, text: str) -> List[str]:
        splits = []
        input_ids = self._tokenizer.encode(text)
        start_idx = 0
        cur_idx = min(start_idx + self._chunk_size, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        while start_idx < len(input_ids):
            splits.append(self._tokenizer.decode(chunk_ids))
            start_idx += self._chunk_size - self._chunk_overlap
            cur_idx = min(start_idx + self._chunk_size, len(input_ids))
            chunk_ids = input_ids[start_idx:cur_idx]
        return splits
    
    def create_docs(self, texts: List[str], metadatas: Optional[List[dict]]=None) -> List[Document]:
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                new_doc = Document(
                    page_content=chunk, metadata=copy.deepcopy(_metadatas[i])
                )
                documents.append(new_doc)
            
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.create_docs(texts, metadatas=metadatas)
    
    def doc_renum_id(self, documents: List[Document], prefix="", suffix="") -> List[Document]:
        new_documents = copy.deepcopy(documents)
        for doc_num, doc in enumerate(new_documents):
            doc.metadata["id"] = prefix + str(doc_num) + suffix
        return new_documents