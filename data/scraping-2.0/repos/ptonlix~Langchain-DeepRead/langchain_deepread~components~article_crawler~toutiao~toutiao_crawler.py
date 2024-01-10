from langchain_deepread.components.article_crawler.base.base import (
    ArticleCrawlerWebDriver,
)


class TouTiaoCrawler(ArticleCrawlerWebDriver):
    def __init__(
        self,
        title_xpath: str = '//div[@class="article-content"]/h1',
        author_xpath: str = '//span[@class="name"]/a',
        content_xpath: str = '//article[contains(@class,"syl-article-base")]',
        source: str = "今日头条",
    ):
        super().__init__(
            title_xpath=title_xpath,
            author_xpath=author_xpath,
            content_xpath=content_xpath,
            source=source,
        )


if __name__ == "__main__":
    tt = TouTiaoCrawler()
    art = tt.article_webdriver(
        url="https://www.toutiao.com/article/7312090318093386278/?log_from=972ebfe1cba42_1702520041079"
    )
    print(art)
    tt.refresh()
    print()
    art = tt.article_webdriver(
        url="https://www.toutiao.com/article/7312035756649431561/?log_from=366489bdf1c41_1702520078848"
    )
    print(art)
    tt.refresh()

    print()
    art = tt.article_webdriver(
        url="https://www.toutiao.com/article/7312156043738415655/?log_from=36cf41bc34923_1702520881355"
    )
    print(art)

    tt.refresh()
    art = tt.article_webdriver(
        url="https://www.toutiao.com/article/7311867758944420403/?log_from=ced65f405397e_1702520914751"
    )
    print(art)
