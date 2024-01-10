from langchain_deepread.components.article_crawler.base.base import (
    ArticleCrawlerWebDriver,
)


class SspaiCrawler(ArticleCrawlerWebDriver):
    def __init__(
        self,
        title_xpath: str = '//*[@id="article-title"]',
        author_xpath: str = '//div[@class="ss__user__nickname el-popover__reference"]/span',
        content_xpath: str = '//*[@id="app"]/div[1]/div[1]/article/div[2]/div[1]/div[2]',
        source: str = "少数派",
    ):
        super().__init__(
            title_xpath=title_xpath,
            author_xpath=author_xpath,
            content_xpath=content_xpath,
            source=source,
        )


if __name__ == "__main__":
    tt = SspaiCrawler()
    art = tt.article_webdriver(url="https://sspai.com/post/84928")
    print(art)
    tt.refresh()
    print()
    art = tt.article_webdriver(url="https://sspai.com/post/84976")
    print(art)
    tt.refresh()

    print()
    art = tt.article_webdriver(url="https://sspai.com/post/85007")
    print(art)

    tt.refresh()
    art = tt.article_webdriver(url="https://sspai.com/post/85032")
    print(art)
