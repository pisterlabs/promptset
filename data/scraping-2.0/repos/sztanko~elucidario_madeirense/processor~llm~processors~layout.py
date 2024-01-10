from typing import List, Dict, Optional
import logging
import json
from collections import defaultdict

from processor.llm.processor import Processor
from processor.utils.splitter import split_article
from processor.utils.instructions import load_instructions, make_output_schema_instructions
from models.article import Article
from models import create_list_model
from processor.llm.engines import OpenAIEngine, AIEngine, ClaudeEngine, GPT4Engine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class LayoutProcessor(Processor):
    def __init__(self, engine_str: Optional[str], strict_validation: Optional[bool]=True):
        super().__init__(engine_str=engine_str)
        self.output_schema = make_output_schema_instructions(Article)
        self.output_schema_list = make_output_schema_instructions(Article, as_list=True)
        engine = self.engine or GPT4Engine
        self.simple_article_engine = engine(
            load_instructions("instructions/layout/single_article.txt", output_schema=self.output_schema)
        )
        self.multiple_articles_engine = engine(
            load_instructions("instructions/layout/multiple_articles.txt", output_schema=self.output_schema_list)
        )
        self.partial_article_engine = engine(
            load_instructions("instructions/layout/article_chunk.txt", output_schema=self.output_schema)
        )
        self.strict_validation = strict_validation

    def get_single_article_engine(self) -> AIEngine:
        return self.simple_article_engine

    def get_multiple_articles_engine(self) -> AIEngine:
        return self.multiple_articles_engine

    def get_partial_article_engine(self) -> AIEngine:
        return self.partial_article_engine

    def to_message(self, articles: List[str]) -> str:
        return "\n\n".join(articles)

    def split_articles(self, message: str, threshold: int) -> List[str]:
        return split_article(message, threshold)

    def parse_result(self, result: str) -> Dict:
        res = json.loads(result)
        logging.info("Result is")
        logging.info(res)
        if self.strict_validation:
            article = Article(**res)
            return article.__dict__
        return res

    def parse_chunk_result(self, result: str) -> Dict:
        # In case of articles, it is same as parse_result
        return self.parse_result(result)

    def parse_multiple_result(self, result: str) -> List[Dict]:
        res = json.loads(result)
        results = []
        for a in res["a"]:
            try:
                if self.strict_validation:
                    article = Article(**a)
                    results.append(article.__dict__)
                else:
                    results.append(a)
            except Exception as e:
                # This is not ideal, because we won't be retrying anything, but it's better than nothing
                logging.error(e)
                logging.error(a)
        return results

    def union_dicts(self, chunks: List[Article], field):
        array_of_dicts = [chunk.__dict__.get(field) or {} for chunk in chunks]
        out = defaultdict(list)
        for chunk in array_of_dicts:
            for k, v in chunk.items():
                out[k].append(v)
        return dict([(k, "\n\n".join(out[k])) for k in out])

    def merge_results(self, results: List[Dict]) -> Dict:
        chunks = [Article(**res) for res in results]
        merged_article = Article(
            title=chunks[0].title,
            id=chunks[0].id,
            body="\n\n".join([chunk.body for chunk in chunks]),
            references=list(set([ref for chunk in chunks for ref in chunk.references or []])),
            categories=list(set([cat for chunk in chunks for cat in chunk.categories or []])),
            freguesias=list(set([freg for chunk in chunks for freg in chunk.freguesias or []])),
            years=self.union_dicts(chunks, "years"),
            locations=self.union_dicts(chunks, "locations"),
            people=self.union_dicts(chunks, "people"),
        )
        return merged_article.__dict__
