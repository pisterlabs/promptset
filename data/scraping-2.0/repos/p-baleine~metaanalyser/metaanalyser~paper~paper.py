import arxiv
import datetime
import logging
import re
import tempfile
from collections import Counter
from langchain.base_language import BaseLanguageModel
from langchain.utilities import SerpAPIWrapper
from pdfminer.high_level import extract_text
from pydantic import BaseModel
from tqdm.auto import tqdm
from typing import List, Optional

from ..memory import memory
from .arxiv_categories import CATEGORY_NAME_ID_MAP


logger = logging.getLogger(__name__)


class Citation(BaseModel):

    title: str
    snippet: str


class GoogleScholarItem(BaseModel):

    result_id: str
    title: str
    link: str
    nb_cited: int
    citations: List[Citation]

    @property
    def mla_citiation(self) -> str:
        mla = [c for c in self.citations if c.title == 'MLA']

        if mla:
            return mla[0]

    @classmethod
    def from_google_scholar_result(cls, result):
        result_id = result["result_id"]
        link = result["link"] if "link" in result else ""
        nb_cited = (
            result["inline_links"]["cited_by"]["total"]
            if "cited_by" in result["inline_links"] else 0
        )
        citations = [
            Citation(title=c["title"], snippet=c["snippet"]) for c in
            fetch_google_scholar_cite(result_id)["citations"]
        ]

        return cls(
            result_id=result_id,
            title=result["title"],
            link=link,
            nb_cited=nb_cited,
            citations=citations,
        )


class Paper(BaseModel):
    """論文を表す、Google Scholar で得られる情報に追加して doi や要約などのフィールドを持つ

    NOTE: serpapi 以外をソースにすることも考えられるが、今は Paper の出自は serpapi の検索結果に限定する
    """

    citation_id: int
    google_scholar_item: GoogleScholarItem
    entry_id: str
    summary: str
    published: datetime.datetime
    primary_category: str
    categories: List[str]
    text: str
    doi: Optional[str]

    @property
    def google_scholar_result_id(self):
        return self.google_scholar_item.result_id

    @property
    def title(self) -> str:
        return self.google_scholar_item.title

    @property
    def link(self) -> str:
        return self.google_scholar_item.link

    @property
    def nb_cited(self) -> int:
        return self.google_scholar_item.nb_cited

    @property
    def citations(self) -> str:
        return self.google_scholar_item.citations

    @property
    def mla_citiation(self) -> str:
        return self.google_scholar_item.mla_citiation

    @classmethod
    def from_google_scholar_result(cls, citation_id, result):
        google_scholar_item = GoogleScholarItem.from_google_scholar_result(result)
        arxiv_result = fetch_arxiv_result(google_scholar_item.link)

        def get_category(c):
            if c not in CATEGORY_NAME_ID_MAP:
                logger.warning(f'Category {c} is not found in CATEGORY_NAME_ID_MAP.')
                return None
            return CATEGORY_NAME_ID_MAP[c]

        primary_category = get_category(arxiv_result.primary_category)
        categories = [
            c for c in [get_category(c) for c in arxiv_result.categories]
            if c
        ]

        return cls(
            citation_id=citation_id,
            google_scholar_item=google_scholar_item,
            entry_id=arxiv_result.entry_id,
            summary=arxiv_result.summary,
            published=arxiv_result.published,
            primary_category=primary_category,
            categories=categories,
            doi=arxiv_result.doi,
            text=get_text_from_arxiv_search_result(arxiv_result),
        )

    def _repr_html_(self):
        def get_category_string():
            # 基本的に categories の先頭が primary_category らしい
            if not self.categories:
                return ""

            result = f"<span style=\"font-weight: bold\">{self.categories[0]}</span>"

            if len(self.categories) == 1:
                return result

            return f"{result}; " + "; ".join([c for c in self.categories[1:]])

        return (
            "<div>"
            f"  Title:&nbsp;<a href=\"{self.link}\" target=\"_blank\">{self.title}</a><br/>"
            f"  引用:&nbsp;[{self.citation_id}] {self.mla_citiation.snippet}<br/>"
            f"  被引用数:&nbsp;{self.nb_cited}<br/>"
            f"  発行日:&nbsp;{self.published}<br/>"
            f"  カテゴリ:&nbsp;{get_category_string()}<br/>"
            f"  要約:&nbsp;{self.summary}<br/>"
            "</div>"
        )


def search_on_google_scholar(
        query: str,
        approved_domains: List[str] = ["arxiv.org"],
        n: int = 10,
) -> List[Paper]:
    """query で SerpApi の Google Scholar API に問合せた結果を返す。
    approved_domains に指定されたドメインの論文のみを対象とする。
    最大 n に指定された件数を返却する。
    """

    def fetch(start=0):
        def valid_item(i):
            if "link" not in i:
                return False

            domain = re.match(r"https?://([^/]+)", i["link"])

            if not domain or domain.group(1) not in approved_domains:
                return False

            return True

        # FIXME: 検索結果に arxiv の文献をなるべく多く含めたいため検索クエリを弄っている
        actual_query = " ".join([query, "arxiv"]) if "arxiv" not in query.lower() else query
        search_result = fetch_google_scholar(actual_query, start)

        return [i for i in search_result if valid_item(i)]

    result = []
    start = 0

    while len(result) < n:
        # FIXME: 今のままだとそもそも検索結果が全体で n 件以下の場合に無限ループになってしまう
        result += fetch(start)
        start += 10

    logger.info("Collecting details...")

    return [
        Paper.from_google_scholar_result(id, i)
        for id, i in tqdm(enumerate(result[:n], start=1))
    ]


def get_categories_string(papers: List[Paper], n: int = 3) -> str:
    categories = Counter(sum([p.categories for p in papers], []))
    common = categories.most_common(n)

    if not common:
        return "Artifical Intelligence"

    if len(common) == 1:
        return common[0][0]

    if len(common) == 2:
        return " and ".join([c[0] for c in common])

    *lst, last = common

    return ", ".join([c[0] for c in lst]) + f" and {last[0]}"


def get_abstract_with_token_limit(
        model: BaseLanguageModel,
        papers: List[Paper],
        limit: int,
        separator: str = "\n",
) -> str:
    def get_summary(paper: Paper):
        summary = paper.summary.replace("\n", " ")
        return f"""
Title: {paper.title}
citation_id: {paper.citation_id}
Summry: {summary}
"""

    summaries = []
    total_num_tokens = 0
    idx = 0

    while idx < len(papers):
        summary = get_summary(papers[idx])
        num_tokens = model.get_num_tokens(summary)

        if total_num_tokens + num_tokens > limit:
            break

        summaries.append(summary)
        total_num_tokens += num_tokens
        idx += 1

    result = separator.join(summaries).strip()

    logger.info(
        f'Number of papers: {len(summaries)}, '
        f'number of tokens: {total_num_tokens}, text: {result[:100]}...'
    )

    return result


@memory.cache
def fetch_google_scholar(query: str, start: int) -> dict:
    logger.info(f"Looking for `{query}` on Google Scholar, offset: {start}...")
    serpapi = SerpAPIWrapper(params={
        "engine": "google_scholar",
        "gl": "us",
        "hl": "en",
        "start": start,
    })
    return serpapi.results(query)["organic_results"]


@memory.cache
def fetch_google_scholar_cite(google_scholar_id: str) -> dict:
    serpapi = SerpAPIWrapper(params={"engine": "google_scholar_cite"})
    return serpapi.results(google_scholar_id)


@memory.cache
def fetch_arxiv_result(arxiv_abs_link: str) -> arxiv.Result:
    m = re.match(r"https://arxiv\.org/abs/(.+)", arxiv_abs_link)
    assert m is not None, f"{arxiv_abs_link} should be a arxiv link"
    arxiv_id = m.group(1)
    return next(arxiv.Search(id_list=[arxiv_id]).results())


@memory.cache
def get_text_from_arxiv_search_result(
        arxiv_search_result: arxiv.Result
) -> str:
    with tempfile.TemporaryDirectory() as d:
        file_path = arxiv_search_result.download_pdf(dirpath=d)
        return extract_text(file_path)
