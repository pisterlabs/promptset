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

from langchain.document_loaders import __all__ as loaders

from document_loaders.AlitaCSVLoader import AlitaCSVLoader
from document_loaders.AlitaExcelLoader import AlitaExcelLoader
from document_loaders.AlitaDirectoryLoader import AlitaDirectoryLoader
from document_loaders.AlitaGitRepoLoader import AlitaGitRepoLoader
from document_loaders.AlitaConfluenceLoader import AlitaConfluenceLoader


ex_classes = {
    'DirectoryLoader': AlitaDirectoryLoader,
    'CSVLoader': AlitaCSVLoader,
    'ExcelLoader': AlitaExcelLoader,
    'GitLoader': AlitaGitRepoLoader,
    'ConfluenceLoader': AlitaConfluenceLoader
}

class LoaderInterface:
    def __init__(self, loader_name, **kwargs):    
        self.loader = LoaderInterface.get_loader_cls(loader_name)(**kwargs)
    
    @staticmethod
    def get_loader_cls(loader_name):
        if loader_name in ex_classes:
            loader = ex_classes[loader_name]
        elif loader_name in loaders:
            loader = getattr(
                __import__("langchain.document_loaders", fromlist=[loader_name]), loader_name
            )
        else:
            loader = getattr(
                __import__("langchain.document_loaders", fromlist=[loader_name]), 'TextLoader'
            )
        return loader

    def load(self, *args, **kwargs):
        return self.loader.load(*args, **kwargs)

    def load_and_split(self, *args, **kwargs):
        return self.loader.load_and_split(*args, **kwargs)
    
    def lazy_load(self, *args, **kwargs):
        return self.loader.lazy_load(*args, **kwargs)



def get_data(loader, load_params):
    if not load_params:
        load_params = {}
    try:
        doc_loader = loader.lazy_load(**load_params)
    except (NotImplementedError, TypeError):
        doc_loader = loader.load(**load_params)
    for _ in doc_loader:
        yield _        
    return


def loader(loader_name, loader_params, load_params):
    if loader_params.get('loader_cls'):
        loader_cls = LoaderInterface.get_loader_cls(loader_params.get('loader_cls'))
        loader_params['loader_cls'] = loader_cls   

    loader = LoaderInterface(loader_name, **loader_params)
    for document in get_data(loader, load_params):
        yield document