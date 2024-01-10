from langchain_deepread.components.article_crawler.base.base import (
    ArticleCrawlerWebDriver,
)


class TmtCrawler(ArticleCrawlerWebDriver):
    def __init__(
        self,
        title_xpath: str = '//div[@class="post_part_title"]/h2',
        author_xpath: str = '//div[@class="author_box"]/a[2]/p',
        content_xpath: str = '//div[@class="_bottom"]',
        source: str = "钛媒体",
    ):
        super().__init__(
            title_xpath=title_xpath,
            author_xpath=author_xpath,
            content_xpath=content_xpath,
            source=source,
        )


if __name__ == "__main__":
    tt = TmtCrawler()
    art = tt.article_webdriver(url="https://www.tmtpost.com/6834444.html")
    print(art)
    tt.refresh()
    print()
    art = tt.article_webdriver(url="https://www.tmtpost.com/6834942.html")
    print(art)
    tt.refresh()

    print()
    art = tt.article_webdriver(url="https://www.tmtpost.com/6835079.html")
    print(art)
    tt.refresh()

    print()
    art = tt.article_webdriver(url="https://www.tmtpost.com/6834941.html")
    print(art)
