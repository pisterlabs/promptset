from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Union

import bs4
from bs4 import ResultSet, Tag
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.messages import BaseMessageChunk
from pydantic import BaseModel

from insightbeam.common import Article
from insightbeam.config import Configuration

_logger = logging.getLogger(__name__)


class Interpreter:
    _chat_model: BaseChatModel
    _fail_token = "IGNORE"

    _gen_analysis_sys_msg = """You analyze articles and help the user determine the main subject matter the article
    is talking about along with the view points made and supporting arguments for that view point. The report you
    provide should be in the following format:

     <analysis>
        <subject>[subject goes here]</subject>
        <view-points>
            <view-point>
                <point>[The point being made]</point>
                <arguments>
                    <argument>[supporting argument that supports the point]</argument>
                </arguments>
            </view-point>
        </view-points>
     </analysis>

     The subject, points and arguments included in the report should be easily searchable in the original article.
     """

    _gen_analysis_template = """
    Article: {article}

    Report:
    """

    _gen_counter_sys_msg = """Given a subject, points made about the subject and related articles, identify
    which, if any that provide countering/opposite points. Your response should be in the following format:
    <analysis>
        <counters>
            <counter>
                <original>One of the original view points being apposed/countered</original>
                <other>The opposing/counter view point being presented</other>
                <article-url>article_url for counter-view-point goes here</article-url>
            </counter>
        </counters>
    </analysis>

    If any of the rules below fail, simply respond with only: `{fail_token}` as the entire response message.
    Rules:
    * Subject should be a non empty string.
    * Under the `Points` section, there should be one to many points each starting with `*` and separated by newlines.
    * There will be a section called `Related` under which each related article will be provided sepearated by two
    newlines.
    * Each article provided will have an `article_url` property whose value should be a non null and non empty string.
    * Each article provided will have an content` property which value should be a non null and non empty string.

    Only include counter/opposing views in the analysis any that aren't can be ignored.
    """.format(
        fail_token=_fail_token
    )

    _gen_counter_template = """
    Subject:
    {subject}

    Points:
    {points}

    Related:
    {related}
    """

    _point_template = "* {point}"
    _related_article_template = "article_url: {url}\ncontent: {content}"
    _sub_analysis_err_msg_header = "There was an error retrieving the sub analysis"
    _sub_analysis_err_msg_fmt = "{header}, error: [{error}]"

    def __init__(self, cfg: Configuration):
        self._chat_model = ChatOpenAI(
            temperature=0.2,
            openai_api_key=cfg.openai_api_key,
            request_timeout=cfg.dep_call_timeout,
            model="gpt-3.5-turbo-16k",
        )

    def _sub_analysis(self, item: Article) -> BaseMessageChunk:
        _logger.info("Generating analysis for (title) (%s)", item.title)
        return self._chat_model.invoke(
            [
                SystemMessage(content=self._gen_analysis_sys_msg),
                HumanMessage(
                    content=self._gen_analysis_template.format(article=item.content)
                ),
            ]
        )

    def counter_analysis(
        self,
        url: str,
        article_analysis: Analysis,
        relevant: List[Article],
    ) -> ArticleAnalysis:
        points = "\n".join(
            [
                self._point_template.format(point=vp.point)
                for vp in article_analysis.view_points
            ]
        )
        related = "\n\n".join(
            [
                self._related_article_template.format(url=rel.url, content=rel.content)
                for rel in relevant
            ]
        )
        msg = self._gen_counter_template.format(
            subject=article_analysis.subject, points=points, related=related
        )
        try:
            opposing_view = self._chat_model.invoke(
                [
                    SystemMessage(content=self._gen_counter_sys_msg),
                    HumanMessage(content=msg),
                ]
            )

            if opposing_view.content != self._fail_token:
                analysis = CounterAnalysis.parse_xml(opposing_view.content)
                error = None
            else:
                raise ValueError("Interpreter returned fail token")
        except Exception as e:
            analysis = None
            error = str(e)

        return ArticleAnalysis(
            article_url=url,
            analysis=article_analysis,
            counter=analysis,
            error=error,
        )

    def analyze(self, items: List[Article]) -> List[ArticleAnalysis]:
        sub_analyses: Dict[str, str] = dict()
        with ThreadPoolExecutor(max_workers=len(items)) as tpe:
            analysis_tasks = {
                tpe.submit(self._sub_analysis, item): item.url for item in items
            }

            for analysis_task in as_completed(analysis_tasks):
                url = analysis_tasks[analysis_task]
                try:
                    response: BaseMessageChunk = analysis_task.result()
                    sub_analyses[url] = response.content
                except Exception as e:
                    msg = self._sub_analysis_err_msg_fmt.format(
                        header=self._sub_analysis_err_msg_header, error=e
                    )
                    sub_analyses[url] = msg
            return self._process_raw_analysis(sub_analyses)

    def _process_raw_analysis(self, analyses: Dict[str, str]) -> List[ArticleAnalysis]:
        processed_analyses = list()
        for url, analysis in analyses.items():
            try:
                if re.match(self._sub_analysis_err_msg_header, analysis):
                    raise ValueError(analysis)

                parsed_analysis = Analysis.parse_xml(analysis)
                error = None
            except Exception as e:
                parsed_analysis = None
                error = str(e)

            processed_analyses.append(
                ArticleAnalysis(article_url=url, analysis=parsed_analysis, error=error)
            )

        return processed_analyses


class XmlParseNode:
    @classmethod
    def _get_tag(cls, tagname: str, soup: bs4.BeautifulSoup | Tag) -> Tag:
        """
        :raise ValueError: When the tag name cannot be found in the soup/tag provided
        """
        result = soup.find(tagname)
        if result is not None and isinstance(result, Tag):
            return result
        else:
            raise ValueError(f"Could not find {tagname} node")


class ViewPoint(BaseModel, XmlParseNode):
    point: str
    arguments: List[str]

    @classmethod
    def parse_xml(cls, content: Tag) -> ViewPoint:
        point_node = cls._get_tag("point", content)
        arguments_node = cls._get_tag("arguments", content)

        point = point_node.get_text()
        argument_nodes: ResultSet[Tag] = arguments_node.find_all("argument")
        arguments = [arg.get_text() for arg in argument_nodes]

        return cls(point=point, arguments=arguments)


class Analysis(BaseModel, XmlParseNode):
    subject: str
    view_points: List[ViewPoint]

    @classmethod
    def parse_xml(cls, content: str) -> Analysis:
        content_soup = bs4.BeautifulSoup(content, features="lxml")
        analysis_node = cls._get_tag("analysis", content_soup)

        subject_node = cls._get_tag("subject", analysis_node)
        viewpoints_node = cls._get_tag("view-points", analysis_node)

        subject = subject_node.get_text()
        viewpoint_nodes = viewpoints_node.find_all("view-point")
        view_points = [
            ViewPoint.parse_xml(view_point) for view_point in viewpoint_nodes
        ]
        return cls(subject=subject, view_points=view_points)


class Counter(BaseModel, XmlParseNode):
    article_url: str
    original_view_point: str
    counter_view_point: str

    @classmethod
    def parse_xml(cls, content: Tag) -> Counter:
        article_url = cls._get_tag("article-url", content).get_text()
        original = cls._get_tag("original", content).get_text()
        counter = cls._get_tag("other", content).get_text()

        return cls(
            article_url=article_url,
            original_view_point=original,
            counter_view_point=counter,
        )


class CounterAnalysis(BaseModel, XmlParseNode):
    counters: List[Counter]

    @classmethod
    def parse_xml(cls, content: str) -> CounterAnalysis:
        content_soup = bs4.BeautifulSoup(content, features="lxml")
        analysis_node = cls._get_tag("analysis", content_soup)
        counters_node = cls._get_tag("counters", analysis_node)
        counter_nodes = counters_node.find_all("counter")
        counters = [Counter.parse_xml(counter) for counter in counter_nodes]
        return cls(counters=counters)


class ArticleAnalysis(BaseModel):
    article_url: str
    analysis: Union[Analysis, None] = None
    counter: Union[CounterAnalysis, None] = None
    error: Union[str, None] = None
