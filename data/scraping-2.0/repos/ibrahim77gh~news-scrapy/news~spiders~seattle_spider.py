import scrapy
from scrapy import settings
from ..items import NewsItem
import time
import openai

class SeattleSpider(scrapy.Spider):
    def parse_story(self, response):
        items = NewsItem()
        title = response.css('.entry-title::text')[0].extract().strip()
        content = response.css('p:not(.recirculation-widget--item-text)::text').extract()
        content_string = '\n'.join(content)

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

class PoliticsSpider(SeattleSpider):
    name = 'politics'
    start_urls = [
        'https://www.seattletimes.com/seattle-news/politics/'
    ]

    custom_settings = {
        'ITEM_PIPELINES': {
            "news.pipelines.PoliticsPipeline": 300,
        }
    }

    def parse(self, response):
        stories = response.css('.top-story-title a::attr(href)').extract()
        other_stories = response.css('.results-story-title a::attr(href)').extract() 
        stories.extend(other_stories)
        for story_url in stories:
            yield scrapy.Request(story_url, callback=self.parse_story)
            time.sleep(0.5)  # Add a delay of 1 second


class SportsSpider(SeattleSpider):
    name = "sports"
    start_urls = [
        "https://www.seattletimes.com/sports/"
    ]
    custom_settings = {
        'ITEM_PIPELINES': {
            "news.pipelines.SportsPipeline": 300,
        }
    }

    def parse(self, response):
        # .top-stories-list-item a
        stories = response.css('.show a::attr(href)').extract()
        for story_url in stories:
            yield scrapy.Request(story_url, callback=self.parse_story)
            time.sleep(0.5)  # Add a delay of 1 second


class BusinessSpider(SeattleSpider):
    name = "business"
    start_urls = [
        "https://www.seattletimes.com/business/"
    ]
    custom_settings = {
        'ITEM_PIPELINES': {
            "news.pipelines.BusinessPipeline": 300,
        }
    }
    def parse(self, response):
        # .top-stories-list-item a
        stories = response.css('.story-list a::attr(href)').extract()
        for story_url in stories:
            yield scrapy.Request(story_url, callback=self.parse_story)
            time.sleep(0.5)  # Add a delay of 1 second

# .show a

class LocalnewsSpider(SeattleSpider):
    name = "localnews"
    start_urls = [
        'https://www.seattletimes.com/seattle-news/'
    ]
    custom_settings = {
        'ITEM_PIPELINES': {
            "news.pipelines.LocalnewsPipeline": 300,
        }
    }
    def parse(self, response):
        # .top-stories-list-item a
        top_story = response.css('.top-story-title a::attr(href)').extract()
        stories = response.css('.secondary a::attr(href)').extract()
        stories.append(top_story[0])
        for story_url in stories:
            yield scrapy.Request(story_url, callback=self.parse_story)
            time.sleep(0.5)  # Add a delay of 1 second