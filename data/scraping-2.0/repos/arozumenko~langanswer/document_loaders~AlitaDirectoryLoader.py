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
import random
import logging

from pathlib import Path
from typing import Any, Iterator, List, Optional

from langchain.document_loaders import (DirectoryLoader)
from langchain_core.documents import Document
from langchain.document_loaders.directory import _is_visible

from typing import Any, Iterator, List, Optional

from document_loaders.constants import loaders_map


logger = logging.getLogger(__name__)


class AlitaDirectoryLoader(DirectoryLoader):
    def __init__(self, **kwargs):
        self.raw_content = kwargs.get('table_raw_content', False)
        self.page_split = kwargs.get('docs_page_split', False)
        index_inclue_ext = kwargs.get('index_file_exts', '')
        index_exclude_ext = kwargs.get('index_exclude_file_exts', '')
        #
        # Filter documents
        #
        self.index_file_exts = [
            ext.strip()
            for ext in index_inclue_ext.split(",")
            if ext.strip()
        ]
        #
        self.index_exclude_file_exts = [
            ext.strip()
            for ext in index_exclude_ext.split(",")
            if ext.strip()
        ]
        for key in ['table_raw_content', 'docs_page_split', 'index_file_exts', 'index_exclude_file_exts']:
            try:
                del kwargs[key]
            except:
                pass

        super().__init__(**kwargs)
        
    
    def load_file(self, item: Path, path: Path, docs: List[Document], pbar: Optional[Any], retval: Optional[bool] = False):
        """Load a file.

        Args:
            item: File path.
            path: Directory path.
            docs: List of documents to append to.
            pbar: Progress bar. Defaults to None.

        """
        _str_item = str(item)
        _, file_ext = os.path.splitext(_str_item)
        if item.is_file():
            if self.index_file_exts and file_ext not in self.index_file_exts:
                return None
            
            if self.index_exclude_file_exts and file_ext in self.index_exclude_file_exts:
                return None
            
            if _is_visible(item.relative_to(path)) or self.load_hidden:
                try:
                    print(f"Processing file: {_str_item}")
                    if file_ext in loaders_map.keys():
                        if 'raw_content' in loaders_map[file_ext]['kwargs'].keys():
                            loaders_map[file_ext]['kwargs']['raw_content'] = self.raw_content
                        if 'page_split' in loaders_map[file_ext]['kwargs'].keys():
                            loaders_map[file_ext]['kwargs']['page_split'] = self.page_split
                        try:
                            sub_docs = loaders_map[file_ext]['class'](_str_item, **loaders_map[file_ext]['kwargs']).load()
                            if not retval:
                                docs.extend(sub_docs)
                        except Exception as e:
                            try: 
                                sub_docs = self.loader_cls(str(item), **self.loader_kwargs).load()
                                if not retval:
                                    docs.extend(sub_docs)
                            except:
                                pass
                    else:
                        sub_docs = self.loader_cls(str(item), **self.loader_kwargs).load()
                        if not retval:
                            docs.extend(sub_docs)
                            
                except Exception as e:
                    if self.silent_errors:
                        logger.warning(f"Error loading file {str(item)}: {e}")
                    else:
                        raise e
                finally:
                    if pbar:
                        pbar.update(1)
                if retval:
                    print(_)
                    for _ in sub_docs:
                        yield _
    
    def load(self, *args, **kwargs) -> List[Document]:
        return list(self.lazy_load(*args, **kwargs))
    
    def lazy_load(self) -> Iterator[Document]:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: '{self.path}'")
        if not p.is_dir():
            raise ValueError(f"Expected directory, got file: '{self.path}'")

        items = list(p.rglob(self.glob) if self.recursive else p.glob(self.glob))

        if self.sample_size > 0:
            if self.randomize_sample:
                randomizer = (
                    random.Random(self.sample_seed) if self.sample_seed else random
                )
                randomizer.shuffle(items)  # type: ignore
            items = items[: min(len(items), self.sample_size)]

        pbar = None
        if self.show_progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=len(items))
            except ImportError as e:
                logger.warning(
                    "To log the progress of DirectoryLoader you need to install tqdm, "
                    "`pip install tqdm`"
                )
                if self.silent_errors:
                    logger.warning(e)
                else:
                    raise ImportError(
                        "To log the progress of DirectoryLoader "
                        "you need to install tqdm, "
                        "`pip install tqdm`"
                    )

        for i in items:
            for _ in self.load_file(i, p, [], pbar, retval=True):
                yield _

        if pbar:
            pbar.close()
