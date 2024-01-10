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

from langchain.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from typing import List, Optional, Iterator
from charset_normalizer import from_path
import pandas as pd
from json import dumps, loads

try:
    from document_loaders.AlitaTableLoader import AlitaTableLoader
except:
    from AlitaTableLoader import AlitaTableLoader

class AlitaExcelLoader(AlitaTableLoader):
    def read(self):
        df = pd.read_excel(self.file_path, sheet_name=None)
        docs = []
        for key in df.keys():
            if self.raw_content:
                docs.append(df[key].to_string())
            else:
                for record in loads(df[key].to_json(orient='records')):
                    docs.append(record)
        return docs

    def read_lazy(self) -> Iterator[dict]:
        df = pd.read_excel(self.file_path, sheet_name=None)
        for key in df.keys():
            if self.raw_content:
                yield df[key].to_string()
            else:
                for record in loads(df[key].to_json(orient='records')):
                    yield record
        return
