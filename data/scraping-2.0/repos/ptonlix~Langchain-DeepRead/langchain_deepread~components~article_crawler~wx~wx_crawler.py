from langchain_deepread.components.article_crawler.base.base import (
    ArticleCrawlerWebDriver,
)


class WxCrawler(ArticleCrawlerWebDriver):
    def __init__(
        self,
        title_xpath: str = '//*[@id="activity-name"]',
        author_xpath: str = '//*[@id="js_name"]',
        content_xpath: str = '//*[@id="js_content"]',
        source: str = "微信公众号",
    ):
        super().__init__(
            title_xpath=title_xpath,
            author_xpath=author_xpath,
            content_xpath=content_xpath,
            source=source,
        )


if __name__ == "__main__":
    wx = WxCrawler()
    art = wx.article_webdriver(url="https://mp.weixin.qq.com/s/a15uF6hC5aDvZKuVLhwFEQ")
    print(art)
    wx.refresh()
    art = wx.article_webdriver(url="https://mp.weixin.qq.com/s/YWbtLl9Sdfh5jOAgdrEa7A")
    print(art)
    wx.refresh()
