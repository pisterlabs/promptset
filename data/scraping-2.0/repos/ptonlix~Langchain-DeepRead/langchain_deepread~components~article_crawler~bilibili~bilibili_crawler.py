from langchain_deepread.components.article_crawler.base.base import (
    ArticleCrawlerWebDriver,
)


class BiliBiliCrawler(ArticleCrawlerWebDriver):
    def __init__(
        self,
        title_xpath: str = '//div[@class="title-container"]/h1',
        author_xpath: str = '//div[@class="up-name-pannel"]/a',
        content_xpath: str = '//div[@class="article-content"]',
        source="BiliBili",
    ):
        super().__init__(
            title_xpath=title_xpath,
            author_xpath=author_xpath,
            content_xpath=content_xpath,
            source=source,
        )


if __name__ == "__main__":
    bb = BiliBiliCrawler()
    art = bb.article_webdriver(
        url="https://www.bilibili.com/read/cv27882152/?from=category_0"
    )
    print(art)
    bb.refresh()
    print()
    art = bb.article_webdriver(
        url="https://www.bilibili.com/read/cv28436667/?from=category_0"
    )
    print(art)
    bb.refresh()

    print()
    art = bb.article_webdriver(
        url="https://www.bilibili.com/read/cv27731577/?from=category_0"
    )
    print(art)
    bb.refresh()

    print()
    art = bb.article_webdriver(
        url="https://www.bilibili.com/read/cv27731577/?from=category_0"
    )
    print(art)
