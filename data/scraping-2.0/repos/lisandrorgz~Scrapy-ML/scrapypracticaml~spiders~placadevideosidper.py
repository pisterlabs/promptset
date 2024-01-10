import scrapy
from scrapypracticaml.items import GraphicCItem
from fake_useragent import UserAgent
import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter




class PlacadevideosidperSpider(scrapy.Spider):
    name = "placadevideosidper"
    allowed_domains = ["listado.mercadolibre.com.ar", 'www.mercadolibre.com.ar']
    start_urls = ["https://listado.mercadolibre.com.ar/computacion/componentes-pc/placas/placas-video/placa-de-video_NoIndex_True"]
    user_agent = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1"
    download_delay = 2
    cookies_enabled = False
    custom_settings = {
        'FEEDS': {
            'placasdevideo.json': {'format': 'json', 'overwrite': True},
        },
    }
   
   
     

    def parse(self, response):
        item_urls = response.css('div.andes-carousel-snapped__slide a ::attr(href)').getall()
        ua = UserAgent()
        for url in item_urls:
            user_agent = ua.random
            yield response.follow(url, callback=self.parse_item, headers={'User-Agent': user_agent})
        next_page = response.css('li.andes-pagination__button.andes-pagination__button--next a.andes-pagination__link.ui-search-link ::attr(href)').get()
        print("\n\n\nNEXT PAGE ---------------------------->", next_page, "\n\n\n")
        if next_page:
            yield response.follow(next_page, callback=self.parse)

            

    def parse_item(self, response):
        item = GraphicCItem()
        comments_list = response.css('p.ui-review-capability-comments__comment__content::text').getall()
        comments = ' '.join(comments_list)
        title = response.css('h1.ui-pdp-title::text').get()
        dirty_chunks = comments + title
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=5)
        chunks = text_splitter.split_text(dirty_chunks)
        item['url'] = response.url
        item['title'] = response.css('h1.ui-pdp-title::text').get()
        item['price'] = response.css('div.ui-pdp-price__second-line span.andes-money-amount__fraction::text').get()
        item['cuote_cant'] = response.xpath('//p[@class="ui-pdp-color--BLACK ui-pdp-size--MEDIUM ui-pdp-family--REGULAR"]/text()').get()
        item['cuote_price'] = response.css('div.ui-pdp-price__subtitles span.andes-money-amount__fraction::text').get()
        item['stock'] = response.css('span.ui-pdp-review__amount::text').get()
        item['calification'] = response.css('span.ui-pdp-review__rating::text').get()
        item['description'] = response.css('p.ui-pdp-description__content::text').get()
        item['time'] = datetime.datetime.now()
        item['chunks'] = chunks
        yield item


    
    
    


