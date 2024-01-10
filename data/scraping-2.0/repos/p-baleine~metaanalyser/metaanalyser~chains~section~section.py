from langchain.base_language import BaseLanguageModel
from langchain.docstore.document import Document
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.prompts.base import BasePromptTemplate
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from ...paper import (
    Paper,
    get_abstract_with_token_limit,
    get_categories_string,
)
from ..base import (
    SRBaseChain,
    maybe_retry_with_error_output_parser,
)
from ..outline import Outlint
from ..overview import Overview
from .prompt import SECTION_PROMPT


class SRSectionChain(SRBaseChain):

    paper_store: VectorStore
    prompt: BasePromptTemplate = SECTION_PROMPT
    nb_categories: int = 3
    nb_token_limit: int = 1_500
    nb_max_retry: int = 3

    @property
    def input_keys(self) -> List[str]:
        # TODO: 入れ子に対応する
        return [
            "section_idx",
            "query",
            "papers",
            "overview",
            "outline",
            "flatten_sections",
        ]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        input_list = get_input_list(
            self.llm,
            self.paper_store,
            inputs["section_idx"],
            inputs["query"],
            inputs["papers"],
            inputs["overview"],
            inputs["outline"],
            inputs["flatten_sections"],
            self.nb_categories,
            self.nb_token_limit,
        )
        return super()._call(input_list, run_manager=run_manager)

    def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        input_list = get_input_list(
            self.llm,
            self.paper_store,
            inputs["section_idx"],
            inputs["query"],
            inputs["papers"],
            inputs["overview"],
            inputs["outline"],
            inputs["flatten_sections"],
            self.nb_categories,
            self.nb_token_limit,
        )
        return super()._acall(input_list, run_manager=run_manager)


class TextSplit(BaseModel):
    """get_input_list 向けのヘルパークラス
    """

    title: str
    citation_id: int
    text: str

    @classmethod
    def from_paper(cls, paper: Paper) -> "TextSplit":
        return cls(
            title=paper.title,
            citation_id=paper.citation_id,
            text=paper.summary,
        )

    @classmethod
    def from_snippet(cls, snippet: Document) -> "TextSplit":
        return cls(
            title=snippet.metadata["title"],
            citation_id=snippet.metadata["citation_id"],
            text=snippet.page_content,
        )


def get_input_list(
        llm: BaseLanguageModel,
        paper_store: VectorStore,
        section_idx: int,
        query: str,
        papers: List[Paper],
        overview: Overview,
        outline: Outlint,
        flatten_sections,
        nb_categories: int,
        nb_token_limit: int,
        max_paper_store_search_size: int = 100,
):
    section = flatten_sections[section_idx]
    papers_citation_id_map = {p.citation_id: p for p in papers}

    if section.section.citation_ids:
        related_splits = [
            TextSplit.from_paper(papers_citation_id_map[int(citation_id)])
            for citation_id in section.section.citation_ids
        ]
    else:
        # citation_ids が空なら全部を対象とする
        related_splits = [TextSplit.from_paper(p) for p in papers]

    related_splits += [
        TextSplit.from_snippet(snippet) for snippet in
        paper_store.similarity_search(
            f"{section.section.title} {section.section.description}",
            k=max_paper_store_search_size,
        )
    ]

    def get_snippet(split: TextSplit):
        text = split.text.replace("\n", " ")
        return f"""
Title: {split.title}
citation_id: {split.citation_id}
Text: {text}
"""

    snippets = []
    total_num_tokens = 0
    idx = 0

    while idx < len(related_splits):
        split = related_splits[idx]
        snippet_text = get_snippet(split)
        num_tokens = llm.get_num_tokens(snippet_text)

        if total_num_tokens + num_tokens > nb_token_limit:
            break

        snippets.append(snippet_text)
        total_num_tokens += num_tokens
        idx += 1

    return [{
        "query": query,
        "title": overview.title,
        "overview": overview,
        "section_title": section.section.title,
        "section_description": section.section.description,
        "section_level": section.level,
        "md_title_suffix": "#" * section.level,
        "outline": outline,
        "categories": get_categories_string(papers, nb_categories),
        "snippets": "\n".join(snippets).strip(),
    }]
