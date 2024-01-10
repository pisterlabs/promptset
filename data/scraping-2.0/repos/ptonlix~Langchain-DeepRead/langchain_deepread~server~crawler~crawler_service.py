from injector import inject, singleton
from langchain_deepread.components.article_crawler.article_crawler_component import (
    ArticleCrawlerComponent,
)
from langchain_deepread.components.article_crawler import Article
from langchain_deepread.components.article_crawler.exceptions import (
    NotSupportDomainException,
)
import logging

logger = logging.getLogger(__name__)


@singleton
class CrawlerService:
    @inject
    def __init__(
        self,
        article_crawler_component: ArticleCrawlerComponent,
    ) -> None:
        self.article_crawler_component = article_crawler_component

    def crawler(self, url: str) -> Article | None:
        try:
            return self.article_crawler_component.exec(url)
        except NotSupportDomainException:
            logging.error("Crawler sites not supported")
        except Exception as e:
            logging.exception(e)
        return None
