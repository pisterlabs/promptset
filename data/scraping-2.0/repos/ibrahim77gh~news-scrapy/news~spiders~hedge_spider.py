# .Article_title___TC6d a
from ..items import NewsItem
import time
import openai
import scrapy

class HedgeSpider(scrapy.Spider):
    name = 'hedge'
    start_urls = [
        'https://zerohedge.com/'
    ]

    custom_settings = {
        'ITEM_PIPELINES': {
            "news.pipelines.HedgePipeline": 300,
        }
    }

    def parse(self, response):
        # Select all <div> tags with class="package"
        main_url = 'https://zerohedge.com'
        stories = []
        urls = response.css('.Article_title___TC6d a::attr(href)').extract()
        for url in urls:
            story_url = main_url + url
            yield scrapy.Request(story_url, callback=self.parse_story)
            time.sleep(0.5)  # Add a delay of 1 second


        # .NodeBody_container__eeFKv p::text

    def parse_story(self, response):
        items = NewsItem()
        content = response.css('.NodeBody_container__eeFKv p::text').extract()
        content_string = ' \n '.join(content)
        if len(content_string) > 16000:
            content_string = content_string[:16000]
        
        prompt = f'Please summarize this article into 4 sentences without adding or inferring anything: "{content_string}"'
        try:
            completion = openai.Completion.create(
                engine='text-davinci-003',  # text-ada-001
                prompt=prompt,
                max_tokens=200,
                temperature=1
            )
            items['url'] = response.request.url 
            items['content'] = completion["choices"][0]["text"].strip()
        except Exception as e:
            return

        yield items