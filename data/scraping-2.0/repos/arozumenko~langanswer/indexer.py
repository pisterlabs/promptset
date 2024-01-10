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

import os

from typing import Optional

from langchain_core.documents import Document
from interfaces.loaders import loader
from interfaces.kwextractor import KWextractor
from interfaces.splitters import Splitter
from json import dumps
from interfaces.llm_processor import get_embeddings, summarize, get_model, get_vectorstore, add_documents

import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk.downloader
nltk_target = "./nltk_data"
os.makedirs(nltk_target, exist_ok=True)
nltk.downloader._downloader._download_dir = nltk_target
nltk.data.path = [nltk_target]

nltk.download('punkt', download_dir=nltk_target)


def main(dataset: str, library:str, 
         loader_name: str, loader_params: dict, load_params: Optional[dict],
         embedding_model: str,
         embedding_model_params: dict,
         kw_plan: Optional[str], kw_args: Optional[dict],
         splitter_name: Optional[str] = 'chunks', 
         splitter_params: Optional[dict] = {},
         document_processing_prompt: Optional[str] = None,
         chunk_processing_prompt: Optional[str] = None,
         ai_model: Optional[str] = None,
         ai_model_params: Optional[dict] = {},
         vectorstore: Optional[str] = None,
         vectorstore_params: Optional[dict] = {}):
         
    embedding = get_embeddings(embedding_model, embedding_model_params)
    vectorstore = get_vectorstore(vectorstore, vectorstore_params, embedding_func=embedding)
    kw_extractor = KWextractor(kw_plan, kw_args)
    llmodel = get_model(ai_model, ai_model_params)
    for document in loader(loader_name, loader_params, load_params):
        if document_processing_prompt:
            document = summarize(llmodel, document, document_processing_prompt)
            print("Summary: ", document.metadata.get('document_summary', ''))
        if kw_extractor.extractor:
            document.metadata['keywords'] = kw_extractor.extract_keywords(
                document.metadata.get('document_summary', '') + '\n' + document.page_content
            )
            print("Keywords: ", document.metadata['keywords'])
        splitter = Splitter(**splitter_params)
        for index, document in enumerate(splitter.split(document, splitter_name)):
            print("Chunk: ", document.page_content)
            if chunk_processing_prompt:
                document = summarize(llmodel, document, chunk_processing_prompt, metadata_key='chunk_summary')
                if splitter_params.get('kw_for_chunks') and kw_extractor.extractor and document.metadata.get('chunk_summary'):
                    chunk_keywords = kw_extractor.extract_keywords(document.metadata.get('chunk_summary', '') + '\n' + document.page_content)
                    if chunk_keywords:
                        document.metadata['keywords'] = list(set(document.metadata['keywords']).union(chunk_keywords))
            _documents = []
            if document.metadata.get('keywords'):
                _documents.append(
                    Document(page_content=', '.join(document.metadata['keywords']), 
                             metadata={'source': document.metadata['source'], 'type': 'keywords', 'library': library, 'source_type': loader_name, 'dataset': dataset})
                             )
            if document.metadata.get('document_summary'):
                _documents.append(
                    Document(page_content=document.metadata['document_summary'], 
                             metadata={'source': document.metadata['source'], 'type': 'document_summary', 'library': library, 'source_type': loader_name, 'dataset': dataset})
                    )
            if document.metadata.get('og_data'):
                _documents.append(
                    Document(page_content=document.page_content, 
                             metadata={'source': document.metadata['source'], 'type': 'document_summary', 'library': library, 'source_type': loader_name, 'dataset': dataset})
                    )
                _documents.append(
                    Document(page_content=dumps(document.metadata['og_data']), 
                             metadata={'source': document.metadata['source'], 'type': 'data', 'library': library, 'source_type': loader_name, 'dataset': dataset})
                    )
                _documents.append(
                    Document(page_content=', '.join(document.metadata['columns']), 
                             metadata={'source': document.metadata['source'], 'type': 'keywords', 'library': library, 'source_type': loader_name, 'dataset': dataset})
                    )
            elif document.metadata.get('chunk_summary'):
                _documents.append(
                    Document(page_content=document.metadata['chunk_summary'] + '\n\n' + document.page_content, 
                             metadata={'source': document.metadata['source'], 'type': 'data', 'library': library, 'source_type': loader_name, 'dataset': dataset, 'chunk_index': index})
                             )
            else:
                _documents.append(
                    Document(page_content=document.page_content,
                             metadata={'source': document.metadata['source'], 'type': 'data', 'library': library, 'source_type': loader_name, 'dataset': dataset, 'chunk_index': index})
                )
            # print(_documents)
            add_documents(vectorstore=vectorstore, documents=_documents)
        vectorstore.persist()
    return 0


# Logic is the following:
# 1. Loader and its params to get data 
# 2. Keyword extractor and its params to get keywords (for the whole file)
# 3. Splitter and its params to split data (here we may need to use different ar)
# 4. Keyword extractor and its params to get keywords (for the splitted data)
# 5. Embedder and its params to get embeddings (for the splitted data)
