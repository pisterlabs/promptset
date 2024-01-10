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
from langchain_core.documents import Document
from typing import Optional, Any, List
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)


class Splitter:
    def __init__(self, chunk_size=1000, chunk_overlap=100, separators: Optional[List[str]] = None, 
                 regex_separator: Optional[Any] = None, autodetect_language: bool = True, **kwargs: Any):
        self.languages = [e.value for e in Language]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.autodetect_language = autodetect_language
        self.regex_separator = regex_separator
        self.separators = separators
    
    def autodeted_language(self, filepath: str) -> Optional[Language]:
        _, file_ext = os.path.splitext(filepath)
        file_ext = file_ext.replace('.', '')
        if file_ext in self.languages:
            return Language(file_ext)
        else:
            return None
        
    def split(self, document: Document, splitter_name: Optional[str] = 'chunks'):
        if splitter_name == 'lines':
            return self.line_split(document)
        elif splitter_name == 'paragraphs':
            return self.paragraph_split(document)
        elif splitter_name == 'sentences':
            return self.sentence_split(document)
        elif splitter_name == 'chunks':
            return self.chunk_split(document, separators=self.separators)
        else:
            raise NotImplementedError(f"Splitter {splitter_name} is not implemented yet")
    
    def chunk_split(self, document: Document, separators: Optional[List[str]]):
        language = None
        splitter_params = {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
        }
        if self.autodetect_language:
            language = self.autodeted_language(document.metadata['source'])
        if language:
            splitter = RecursiveCharacterTextSplitter.from_language(language=language, **splitter_params)
        else:
            if self.regex_separator:
                splitter_params['separator'] = self.regex_separator
                splitter_params['is_separator_regex'] = True
            if separators:
                splitter_params['separators'] = self.separators
            elif self.separators:
                splitter_params['separators'] = self.separators
            splitter = RecursiveCharacterTextSplitter(**splitter_params)
        return splitter.create_documents([document.page_content], [document.metadata])

    def line_split(self, document: Document):
        return self.chunk_split(document, separators=['\n'])

    def paragraph_split(self, document: Document):
        return self.chunk_split(document, separators=['\n\n'])
    
    def sentence_split(self, document: Document):
        return self.chunk_split(document, separators=['.'])