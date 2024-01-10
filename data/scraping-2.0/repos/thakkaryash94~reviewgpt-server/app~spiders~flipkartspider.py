import os
from typing import TypeVar, Union
import scrapy
import chromadb
import uuid
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from chromadb import Documents, EmbeddingFunction, Embeddings
# from datetime import datetime

TITLE_CLASS_ID = "_2-N8zT"
REVIEW_TEXT_CLASS_ID = "t-ZTKy"
AUTHOR_CLASS_ID = "_2sc7ZR _2V5EHH"
DATE_CLASS_ID = "_2sc7ZR"
VERIFIED_CLASS_ID = "_2mcZGG"
RATING_CLASS_ID_1 = "_1BLPMq"
RATING_CLASS_ID_2 = "_3LWZlK"

REVIEWS_CLASS_ID = "_27M-vq"
READ_MORE_CLASS_ID = "_1BWGvX"
NEXT_PAGE_CLASS_ID = "_1LKTO3"

# Embeddable = Union[Documents, Images]
Embeddable = Documents
D = TypeVar("D", bound=Embeddable, contravariant=True)

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: D) -> Embeddings:
        # embed the documents somehow
        oembed = OllamaEmbeddings(
        base_url="http://127.0.0.1:11434", model="orca2")
        return oembed.embed_documents(input)


def get_collection(collection_name):
    client = chromadb.HttpClient(host='127.0.0.1', port=8000)
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=MyEmbeddingFunction())
    return collection


class Flipkart(scrapy.Spider):
    name = 'flipkart'
    current_page = 1

    def __init__(self, url=None, *args, **kwargs):
        super(Flipkart, self).__init__(*args, **kwargs)
        self.start_urls = [f"{url}"]

    def start_requests(self):
        yield scrapy.Request(
            url=self.start_urls[0],
            meta={
                "playwright": True,
                "playwright_include_page": True,
                "errback": self.errback,
            },
        )

    async def parse(self, response):
        print(
            f"*************Parsing page {self.current_page} {response.url}****************************")
        print(response.meta)
        page = response.meta["playwright_page"]
        reviews = response.xpath(f'.//div[@class="{REVIEWS_CLASS_ID}"]')
        review_index = 0
        read_more_index = 0
        read_mores = await page.locator(f'.{READ_MORE_CLASS_ID}').all()
        collection = get_collection("reviews")
        for review_item in reviews:
            id = str(uuid.uuid4())[:13]
            rating_text = review_item.xpath(
                f'.//div[contains(@class, "{RATING_CLASS_ID_1}")and contains(@class, "{RATING_CLASS_ID_2}")]/text()').get()
            title = review_item.xpath(
                f'.//p[@class="{TITLE_CLASS_ID}"]/text()').get()
            review_text = review_item.xpath(
                f'.//div[@class="{REVIEW_TEXT_CLASS_ID}"]/div/div/text()').get()
            author = review_item.xpath(
                f'.//p[@class="{AUTHOR_CLASS_ID}"]/text()').get()
            verified = review_item.xpath(
                f'.//p[@class="{VERIFIED_CLASS_ID}"]/span/text()').get()
            created_time = review_item.xpath(
                f'.//p[@class="{DATE_CLASS_ID}"]/text()').get()
            is_read_more = bool(review_item.xpath(
                f'.//div[@class="{REVIEW_TEXT_CLASS_ID}"]/div/span[@class="{READ_MORE_CLASS_ID}"]').get() is not None)
            if is_read_more:
                read_more_item = read_mores[read_more_index]
                await read_more_item.click()
                all = await page.query_selector_all(f'.{REVIEW_TEXT_CLASS_ID}>div>div')
                review_text = await all[review_index].inner_html()
                read_more_index = read_more_index + 1
            review_index = review_index + 1
            # print("==========id=========", id)
            # print("==========rating_text=========", rating_text)
            # print("==========title=========", title)
            # print("==========review_text=========", review_text)
            # print("==========author=========", author)
            # print("==========verified=========", verified)
            # print("==========created_time=========", created_time)
            # docs.append(Document(
            #   page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
            #   metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"})
            # )
            document = f'The review for the product is {review_text}. Written by {author} in {created_time} with {rating_text} star ratings.'
            collection.add(
                documents=document,
                metadatas={
                    "title": title,
                    "author": author,
                    "verified": verified,
                    "rating": float(rating_text),
                    "rating_limit": 5,
                    "created_at": created_time,
                },
                ids=id
            )
            print(f"{id} record inserted")
        pages = response.xpath(f'.//a[@class="{NEXT_PAGE_CLASS_ID}"]')
        next_page_url = pages[0].xpath(f'.//@href').get()
        self.current_page = self.current_page + 1
        if len(pages) == 2:
            next_page_url = pages[1].xpath(f'.//@href').get()
        print(
            f"--------next_page_url---------------{response.urljoin(next_page_url)}")
        # fetch 2 pages
        if next_page_url is not None and self.current_page <= 1:
            yield scrapy.Request(
                url=response.urljoin(next_page_url),
                meta={
                    "playwright": True,
                    "playwright_include_page": True,
                    "dont_filter": True,
                    "errback": self.errback,
                },
                callback=self.parse
            )
        else:
            print('*************No Page Left*************')
        await page.close()

    async def errback(self, failure):
        print('-----------ERROR CALLBACK------------------------')
        page = failure.request.meta["playwright_page"]
        await page.close()
