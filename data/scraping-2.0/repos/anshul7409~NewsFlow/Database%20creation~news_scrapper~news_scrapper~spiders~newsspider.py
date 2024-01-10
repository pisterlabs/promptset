import scrapy
# from .config import Config
# import openai_summarize
from .list_of_topics import Topics
from news_scrapper.items import NewsItem
from .Db_conn import get_collection


class NewsspiderSpider(scrapy.Spider):
    name = "newsspider"
    allowed_domains = ["timesofindia.indiatimes.com"]
    topics = Topics.topics_of_news
    start_urls = ['https://timesofindia.indiatimes.com/topic/'+ topic for topic in topics]
    collection  = get_collection()
    
    # openai_summarizer = openai_summarize.OpenAISummarize(Config.OPENAI_KEY)

    def parse(self, response):
        news_data = response.css('div.uwU81')
        if news_data:
            for news_sample in news_data:
                meta_ = news_sample.css('div.VXBf7')
                meta_text = meta_.css('div.ZxBIG').get()
                text = meta_text[meta_text.find('>') + 1:meta_text.rfind('<')]
                date_time = ''
                srcc = ''
                if '/<!-- -->' in text:
                    date_time_text = text.split('/<!-- -->')
                    date_time = date_time_text[1]
                    srcc = date_time_text[0]
                    if len(date_time_text) == 1 :
                      date_time = date_time_text[0]
                      srcc = ''

                item = NewsItem()
                item['url'] = response.urljoin(news_sample.css('a').attrib['href'])
                item['headline'] = meta_.css('div.fHv_i span::text').get()
                item['Src'] = srcc
                item['date_time'] = date_time
                if item['date_time'] == '' or item['headline'] == '' or item['Src'] == '':
                    item['date_time'] = None
                    item['headline'] = None
                    item['Src'] = None
                yield scrapy.Request(item['url'], callback=self.parse_news_page, meta={'item': item})

    def parse_news_page(self, response):
        item = response.meta['item']
        news_content = response.css('div.JuyWl ::text')
        if news_content:
            item['description'] = ' '.join(news_content.getall())                                   
            item['len'] = len(item['description'])
            yield item

# scrapy crawl newsspider -o news1.csv
# o -> appending , O -> overwriting