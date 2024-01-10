import scrapy
from scrapy_llm_loader.loader import LangChainLoader
from pydantic import BaseModel, Field


class ProductPydanticItem(BaseModel):
    name: str = Field(description="name of the product")
    price: str = Field(description="price of the product")


class ExampleSpider(scrapy.Spider):
    name = "example_spider"
    start_urls = [
        "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"
    ]

    def parse(self, response):
        loader = LangChainLoader(
            item_class=ProductPydanticItem, response=response, crawler=self.crawler
        )
        extracted_data = loader.load_item()
        yield extracted_data.dict()
