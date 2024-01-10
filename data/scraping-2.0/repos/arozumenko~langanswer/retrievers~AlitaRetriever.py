# Copyright (c) 2023 Artem Rozumenko
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from langchain_core.retrievers import BaseRetriever
from typing import Any, Dict, List, Optional
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class AlitaRetriever(BaseRetriever):
    vectorstore: Any  # Instance of vectorstore
    doc_library: str  # Name of the document library
    top_k: int  # Number of documents to return
    page_top_k: int = 1
    weights: Dict[str, float] = {
        'keywords': 0.2,
        'document_summary': 0.5,
        'data': 0.3
    }
    
    class Config:
        arbitrary_types_allowed = True 
        
    def _rerank_documents(self, documents: List[tuple]):
        """ Rerank documents """
        _documents = []
        for (document, score) in documents:
            _documents.append({
                "page_content": document.page_content,
                "metadata": document.metadata,
                "score": score*self.weights[document.metadata['type']]
            })
        return sorted(_documents, key=lambda x: x["score"], reverse=True)
    
    def merge_results(self, input:str, docs: List[dict]):
        results = {}
        for doc in docs:
            if doc['metadata']['source'] not in results.keys():
                results[doc['metadata']['source']] = {'page_content': [], 'metadata': { 'source' : doc['metadata']['source'] }}
                documents = self.vectorstore.similarity_search_with_score(input, filter={'source': doc["metadata"]['source']})
                for (d, score) in documents:
                    if d.metadata['type'] == 'data':
                        results[doc['metadata']['source']]['page_content'].append({"content": d.page_content, "index": d.metadata['chunk_index'], "score": score})
                    elif d.metadata['type'] == 'document_summary':
                        results[doc['metadata']['source']]['page_content'].append({"content": d.page_content, "index": -1, "score": score})
            if len(results.keys()) >= self.top_k:
                break
        _docs = []
        for value in results.values():
            _chunks = sorted(value['page_content'], key=lambda x: x["score"], reverse=True)
            pages = list(map(lambda x: x['content'], _chunks))
            _docs.append(Document(page_content = "\n\n".join(pages[:self.page_top_k]), metadata = value['metadata']))
        return _docs
    
    def get_relevant_documents(
        self,
        input: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        docs = self.vectorstore.similarity_search_with_score(input, filter={'library': self.doc_library})
        docs = self._rerank_documents(docs)
        return self.merge_results(input, docs)