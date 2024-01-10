from langchain_deepread.components.article_crawler.base.base import (
    ArticleCrawlerWebDriver,
)


class KrCrawler(ArticleCrawlerWebDriver):
    def __init__(
        self,
        title_xpath: str = '//*[@id="app"]/div/div[1]/div/div[2]/div[3]/div/div/div/div[1]/div/div[1]/div[1]/div/div/div[1]/div/h1',
        author_xpath: str = '//*[@id="app"]/div/div[1]/div/div[2]/div[3]/div/div/div/div[1]/div/div[1]/div[1]/div/div/div[1]/div/div[1]/a',
        content_xpath: str = '//*[@id="app"]/div/div[1]/div/div[2]/div[3]/div/div/div/div[1]/div/div[1]/div[1]/div/div/div[2]/div',
        source: str = "36Kr",
    ):
        super().__init__(
            title_xpath=title_xpath,
            author_xpath=author_xpath,
            content_xpath=content_xpath,
            source=source,
        )


if __name__ == "__main__":
    kr = KrCrawler()
    art = kr.article_webdriver(url="https://36kr.com/p/2557305478258053")
    print(art)
    kr.refresh()
    art = kr.article_webdriver(url="https://36kr.com/p/2557424818969992")
    print(art)
    kr.refresh()
