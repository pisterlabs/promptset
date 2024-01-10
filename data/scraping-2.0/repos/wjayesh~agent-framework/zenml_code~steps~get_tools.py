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
from knowledge.url import URL

from tools.versioned_vector_store import VersionedVectorStoreTool
import zenml_code.zenml_utils as zenml_utils


@step(enable_cache=True)
def get_tools(
    project_name: str,
    versioned_vector_stores: Dict[str, VectorStore],
    all_urls: Dict[str, List[URL]],
) -> Dict[str, VersionedVectorStoreTool]:
    """Returns all the tools available for each version.

    Args:
        versioned_vector_stores: A dictionary with version as key and VectorStore object as value.
        all_urls: A dictionary with version as key and list of URLs as value.

    Returns:
        A dictionary with version as key and VersionedVectorStoreTool object as value.
    """
    # check if a tool (and in turn, a vector store) already
    # exists for some versions
    # TODO figure out how to get the current pipeline name in step
    existing_tools = zenml_utils.get_existing_tools(pipeline_name="index_creation_pipeline")

    # update the existing vector stores with the new ones
    for version in versioned_vector_stores:
        existing_tools[version] = VersionedVectorStoreTool(
            name=f"{project_name}-{version}",
            vector_store=versioned_vector_stores[version],
            version=version,
            description="Use this tool to answer questions about "
            f"project {project_name} at version {version}.",
            # TODO add more description
            # add the hash of all urls for that version
            # to the tool
            urls=[url.get_hash() for url in all_urls[version]],
        )

    return existing_tools
