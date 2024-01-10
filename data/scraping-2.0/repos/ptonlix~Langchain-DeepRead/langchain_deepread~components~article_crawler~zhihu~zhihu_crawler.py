import re

from langchain_deepread.components.article_crawler.base.base import ArticleCrawlerHTTP


class ZhihuCrawler(ArticleCrawlerHTTP):
    def __init__(
        self,
        title_xpath: str = "//h1[@class='Post-Title']/text()",
        author_xpath: str = "//a[@data-za-detail-view-element_name='User']/text()",
        content_tag: str = "div",
        content_soup: dict = {
            "class_": re.compile(r"RichText ztext Post-RichText(\s\w+)?")
        },
        source="知乎",
    ):
        super().__init__(
            title_xpath=title_xpath,
            author_xpath=author_xpath,
            content_tag=content_tag,
            content_soup=content_soup,
            source=source,
        )


if __name__ == "__main__":
    zhihu = ZhihuCrawler()
    art = zhihu.article_http(url="https://zhuanlan.zhihu.com/p/652879015")
    print(art)
