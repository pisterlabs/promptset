from ..items import NewsItem
import time
import openai
import scrapy

class TribuneSpider(scrapy.Spider):
    name = 'tribune'
    start_urls = [
        'https://www.thenewstribune.com/news/local/'
    ]

    def parse(self, response):
        # Select all <div> tags with class="package"
        stories = []
        main_story = response.css('h1 a::attr(href)').extract()
        stories.append(main_story)
        package_divs = response.css('div.package').extract()

        for package_div in package_divs:
            # Extract the URL from the <a> tag inside the <h3> tag
            url = package_div.css('h3 a::attr(href)').get()
            stories.append(url)

        yield stories


        
