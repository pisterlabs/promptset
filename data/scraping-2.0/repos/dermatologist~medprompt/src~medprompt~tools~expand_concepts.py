"""
 Copyright 2023 Bell Eapen

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import Any, List, Optional, Type
from kink import di
from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun,
                                         CallbackManagerForToolRun)
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from huggingface_hub import snapshot_download
from gensim.models import Word2Vec

REPO_ID = di["expand_conceps_repo_id"]
FILENAME = di["expand_concepts_filename"]

class SearchInput(BaseModel):
    concepts: List[str] = Field()

class ExpandConceptsTool(BaseTool):  # StructuredTool if more than one input type
    """
    Expands a list of concepts to include related concepts.
    """
    name = "expand_concepts"
    description = """
    Expands a list of concepts to include related concepts.
    """
    args_schema: Type[BaseModel] = SearchInput

    def _run(
            self,
            concepts: List[str] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None
            ) -> str:
        model = Word2Vec.load(snapshot_download(repo_id=REPO_ID)+"/"+FILENAME)
        _words = []
        phrases = []
        for concept in concepts:
            try:
                _words.extend(model.wv.most_similar(concept, topn=di["expand_concepts_topn"]))
            except:
                pass
        for (word, score) in _words:
            if score > di["expand_concepts_threshold"]:
                phrases.append(word.replace("_", " "))
        return phrases


    async def _arun(
            self,
            concepts: List[str] = None,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None
            ) -> Any:
        return self._run(concepts, run_manager)