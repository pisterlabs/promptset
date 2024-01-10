#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from typing import Dict, List

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
)
from langchain.vectorstores import FAISS, VectorStore
from zenml import step
import zenml_code.zenml_utils as zenml_utils


@step(enable_cache=True)
def index_generator(documents: Dict[str, List[Document]]) -> Dict[str, VectorStore]:
    """Generates a vector store for each version.

    Args:
        documents: A dictionary with version as key and list of Document objects as value.

    Returns:
        A dictionary with version as key and VectorStore object as value.
    """
    # check if a tool (and in turn, a vector store) already
    # exists for some versions
    existing_vector_stores = zenml_utils.get_existing_tools(versions=documents.keys())
    versioned_vector_stores = {}
    for version in documents:
        versioned_vector_stores[version] = None
        embeddings = OpenAIEmbeddings()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        compiled_texts = text_splitter.split_documents(documents)

        if version in existing_vector_stores:
            vector_store = existing_vector_stores[version]
            # TODO check what function the vector store impl
            # for upserting is called
            vector_store.add_texts(compiled_texts, embeddings)
        else:
            vector_store = FAISS.from_documents(compiled_texts, embeddings)

        versioned_vector_stores[version] = vector_store

    return versioned_vector_stores
