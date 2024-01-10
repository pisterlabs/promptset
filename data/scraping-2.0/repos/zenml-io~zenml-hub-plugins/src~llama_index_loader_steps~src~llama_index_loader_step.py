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


from typing import Dict, List, Union

from langchain.docstore.document import Document
from llama_index import download_loader
from zenml.steps import BaseParameters, step


class LlamaIndexLoaderParameters(BaseParameters):
    """Params for Llama Index Hub loader.

    Attributes:
        loader_name: Name of the loader.
        loader_kwargs: Dictionary of keyword arguments to be passed to the loader.
        load_data_kwargs: Dictionary of keyword arguments to be passed to the load_data method.
    """

    loader_name: str
    authentication_kwargs: Dict[str, str]
    loader_custom_kwargs: Dict[str, Union[str, List[str, int]]]


@step
def llama_index_loader_step(
    params: LlamaIndexLoaderParameters,
) -> List[Document]:
    """Loader for documents downloaded using Llama Index Hub loaders.

    Args:
        params: Parameters for the step.

    Returns:
        List of langchain documents.
    """
    docs_loader = download_loader(params.loader_name)

    docs_reader = docs_loader(**params.authentication_kwargs)
    docs = docs_reader.load_data(**params.loader_custom_kwargs)
    return [doc.to_langchain_format() for doc in docs]
