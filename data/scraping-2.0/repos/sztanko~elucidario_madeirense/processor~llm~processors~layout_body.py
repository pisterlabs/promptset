from typing import List, Dict
import logging
import json
from collections import defaultdict

from processor.llm.processors.layout import LayoutProcessor
from processor.utils.splitter import split_article
from processor.utils.instructions import load_instructions, make_output_schema_instructions
from models.article_body import ArticleBody
from models import create_list_model
from processor.llm.engines import OpenAIEngine, AIEngine, ClaudeEngine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class LayoutBodyProcessor(LayoutProcessor):
    def __init__(self):
        super().__init__()
        self.output_schema = make_output_schema_instructions(ArticleBody)
        self.output_schema_list = make_output_schema_instructions(ArticleBody, as_list=True)
        engine = ClaudeEngine
        self.simple_article_engine = engine(
            load_instructions("instructions/layout/single_article.txt", output_schema=self.output_schema)
        )
        self.multiple_articles_engine = engine(
            load_instructions("instructions/layout/multiple_articles.txt", output_schema=self.output_schema_list)
        )
        self.partial_article_engine = engine(
            load_instructions("instructions/layout/article_chunk.txt", output_schema=self.output_schema)
        )


    def parse_result(self, result: str) -> Dict:
        res = json.loads(result)
        logging.info("Result is")
        logging.info(res)
        article = ArticleBody(**res)
        return article.__dict__

    def parse_multiple_result(self, result: str) -> List[Dict]:
        res = json.loads(result)
        ListModel = create_list_model(ArticleBody)
        articles = ListModel(**res).a
        results = [article.__dict__ for article in articles]
        return results

    def merge_results(self, results: List[Dict]) -> Dict:
        chunks = [ArticleBody(**res) for res in results]
        merged_article = ArticleBody(
            title=chunks[0].title,
            id=chunks[0].id,
            body="\n\n".join([chunk.body for chunk in chunks])
        )
        return merged_article.__dict__
