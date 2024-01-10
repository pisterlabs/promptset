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
from langchain.document_loaders import UnstructuredURLLoader
from zenml import step

from agent.agent import URL


@step(enable_cache=True)
def web_url_loader(all_urls: Dict[str, List[URL]]) -> Dict[str, List[Document]]:
    """Loads documents from a list of URLs for each version.

    Args:
        all_urls: A dictionary with version as key and list of URLs as value.

    Returns:
        A dictionary with version as key and list of Document objects as value.
    """
    documents = {}
    for version in all_urls:
        documents[version] = UnstructuredURLLoader(
            urls=[url.url for url in all_urls[version]]
        ).load()
    
    return documents