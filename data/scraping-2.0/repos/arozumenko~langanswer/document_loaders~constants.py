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

from langchain.document_loaders import (TextLoader, 
        UnstructuredMarkdownLoader,
        UnstructuredPDFLoader, UnstructuredWordDocumentLoader,
        JSONLoader, AirbyteJSONLoader, UnstructuredHTMLLoader, 
        UnstructuredPowerPointLoader, PythonLoader)

from .AlitaCSVLoader import AlitaCSVLoader
from .AlitaExcelLoader import AlitaExcelLoader

loaders_map = {
    '.txt': {
        'class': TextLoader,
        'kwargs': {
            'autodetect_encoding': True
        }
    },
    '.md': {
        'class': UnstructuredMarkdownLoader,
        'kwargs': {}
    },
    '.csv': {
        'class': AlitaCSVLoader,
        'kwargs': {
            'encoding': 'utf-8',
            'raw_content': False
        }
    },
    '.xlsx': {
        'class': AlitaExcelLoader,
        'kwargs': { 
            'raw_content': False
        }
    },
    '.xls': {
        'class': AlitaExcelLoader,
        'kwargs': {
            'raw_content': False
        }
    },
    '.pdf': {
        'class': UnstructuredPDFLoader,
        'kwargs': {}
    },
    '.docx': {
        'class': UnstructuredWordDocumentLoader,
        'kwargs': {}
    },
    '.json': {
        'class': TextLoader,
        'kwargs': {}
    },
    '.jsonl': {
        'class': AirbyteJSONLoader,
        'kwargs': {}
    },
    '.htm': {
        'class': UnstructuredHTMLLoader,
        'kwargs': {}
    },
    '.html': {
        'class': UnstructuredHTMLLoader,
        'kwargs': {}
    },
    '.ppt': {
        'class': UnstructuredPowerPointLoader,
        'kwargs': {}
    },
    '.pptx': {
        'class': UnstructuredPowerPointLoader,
        'kwargs': {}
    },
    '.py': {
        'class': PythonLoader,
        'kwargs': {}
    }
}