from langchain_deepread.components.article_crawler.base.base import (
    ArticleCrawlerWebDriver,
)


class JianShuCrawler(ArticleCrawlerWebDriver):
    def __init__(
        self,
        title_xpath: str = '//*[@id="__next"]/div[1]/div/div/section[1]/h1',
        author_xpath: str = '//*[@id="__next"]/div[1]/div/div/section[1]/div[2]/div/div/div[1]/span[1]/a',
        content_xpath: str = '//*[@id="__next"]/div[1]/div/div/section[1]/article',
        source: str = "简书",
    ):
        super().__init__(
            title_xpath=title_xpath,
            author_xpath=author_xpath,
            content_xpath=content_xpath,
            source=source,
        )


if __name__ == "__main__":
    tt = JianShuCrawler()
    art = tt.article_webdriver(url="https://www.jianshu.com/p/c1761deabebd")
    print(art)
    tt.refresh()
    print()
    art = tt.article_webdriver(url="https://www.jianshu.com/p/0de23a693587")
    print(art)
    tt.refresh()
