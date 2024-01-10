from __future__ import annotations

from typing import Any, List

from ..document import Document

import copy

"""
slack off
"""
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter as RCTS
except ImportError:
    raise ImportError(
        "`langchain` package not found, please install it with "
        "`pip install langchain`"
    )

class RecursiveCharacterTextSplitter(RCTS):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def split(self, pdf_content: Document) -> List[str]:
        source = copy.deepcopy(pdf_content[0].metadata)

        if "page" in source.keys():
            del source["page"]

        source = str(source)

        return self.split_text("\n".join([c.content for c in pdf_content]) + source)