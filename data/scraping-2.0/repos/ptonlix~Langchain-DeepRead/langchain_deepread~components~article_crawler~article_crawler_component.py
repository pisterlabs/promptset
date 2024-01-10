from injector import singleton
from langchain_deepread.components.article_crawler.base import (
    ArticleCrawerBase,
)
from langchain_deepread.components.article_crawler.bilibili import (
    BiliBiliCrawler,
)
from langchain_deepread.components.article_crawler.jianshu import (
    JianShuCrawler,
)
from langchain_deepread.components.article_crawler.kr import (
    KrCrawler,
)
from langchain_deepread.components.article_crawler.sspai import (
    SspaiCrawler,
)
from langchain_deepread.components.article_crawler.tmt import (
    TmtCrawler,
)
from langchain_deepread.components.article_crawler.toutiao import (
    TouTiaoCrawler,
)
from langchain_deepread.components.article_crawler.wx import WxCrawler
from langchain_deepread.components.article_crawler.zhihu import ZhihuCrawler
from langchain_deepread.components.article_crawler.base import Article
from urllib.parse import urlparse
from langchain_deepread.components.article_crawler.exceptions import (
    NotSupportDomainException,
)


@singleton
class ArticleCrawlerComponent:
    def __init__(self):
        ...

    """
    解析URL的域名
    """

    def _extract_domain(self, url: str) -> str:
        # 解析 URL
        parsed_url = urlparse(url)

        # 获取域名
        domain = parsed_url.netloc

        # 去掉 'www' 前缀
        if domain.startswith("www."):
            domain = domain[4:]

        # 去掉域名后缀
        parts = domain.split(".")
        if len(parts) > 1:
            domain = ".".join(parts[:-1])

        return domain

    """
    获取URL解析返回内容
    """

    def exec(self, url: str) -> Article | None:
        domain = self._extract_domain(url)
        crawler: ArticleCrawerBase = None
        art: Article = None
        match domain:
            case "bilibili":
                crawler = BiliBiliCrawler()
                art = crawler.article_webdriver(url)
            case "jianshu":
                crawler = JianShuCrawler()
                art = crawler.article_webdriver(url)
            case "36kr":
                crawler = KrCrawler()
                art = crawler.article_webdriver(url)
            case "sspai":
                crawler = SspaiCrawler()
                art = crawler.article_webdriver(url)
            case "tmtpost":
                crawler = TmtCrawler()
                art = crawler.article_webdriver(url)
            case "toutiao":
                crawler = TouTiaoCrawler()
                art = crawler.article_webdriver(url)
            case "mp.weixin.qq":
                crawler = WxCrawler()
                art = crawler.article_webdriver(url)
            case "zhuanlan.zhihu":
                crawler = ZhihuCrawler()
                art = crawler.article_http(url)
            case _:
                raise NotSupportDomainException

        crawler.refresh()
        return art
