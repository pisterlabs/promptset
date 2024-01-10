import scrapy
import time
from datetime import datetime

from openaicommentstoscrape.items import CommentsItem, TopicItem

# Scrapy spider to crawl site and fetch topic and its comments
class TopicsSpider(scrapy.Spider):
    name = "topics"
    allowed_domains = ["community.openai.com"]
    start_urls = ["https://community.openai.com/"]

    def check_months_old(self, activity):
        formatted_date = datetime.strptime(activity, '%B %d, %Y')
        current_date = datetime.now()
        return (current_date.year - formatted_date.year) * 12 + current_date.month - formatted_date.month

    def parse(self, response):
        # to avoid 429 response
        time.sleep(5)
        # response will load default landing page of https://community.openai.com/
        # for loop will iterate over list of topics and fetch detail of individual item
        for topic in response.xpath('//table/tbody/tr'):
            topic_name = topic.xpath('.//td[1]/span/a/text()').get()
            replies = topic.xpath('.//td[3]/span/text()').get()
            views = topic.xpath('.//td[4]/span/text()').get()
            activity = topic.xpath('.//td[5]/text()').get().strip()
            topic_url = topic.xpath('.//td[1]/span/a').attrib['href']
            # it will crawl for individual topic_url to fetch comments and parse them with parse_comment
            yield response.follow(
                url = topic_url,
                callback=self.parse_comments,
                meta={ 
                    "topic": topic_name,
                    "replies": replies,
                    "views": views,
                    "activity": activity,
                    "topic_url": topic_url
                    }
                )

        # identify how many months old the topic is
        month_difference = self.check_months_old(activity)

        # load next page only if the last topic activity is less than 3 months old
        if month_difference <= 3: 
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            }
            # get next page url from the bottom of the page
            next_page_url = response.xpath('//div[@class="navigation"]//a[@rel="next"]').attrib['href']
            print(next_page_url)
            # load next page
            yield response.follow(url = next_page_url, callback = self.parse, headers=headers)


    # parser to parse comments from topic page
    def parse_comments(self, response):
        topic = response.request.meta["topic"]
        replies = response.request.meta["replies"]
        views = response.request.meta["views"]
        activity = response.request.meta["activity"]
        topic_url = response.request.meta["topic_url"]

        # TopicItem model will hold data and push to db through scrapy pipeline
        topicItem = TopicItem()
        topicItem['topic_name'] = topic
        topicItem['replies'] = replies
        topicItem['views'] = views
        topicItem['activity'] = activity
        topicItem['topic_url'] = topic_url
        yield topicItem

        comments = []
        for topic_body in response.xpath('//div[@class="topic-body crawler-post"]'):
            comments.append({
                'author': topic_body.xpath('.//span[@class="creator"]//span[@itemprop="name"]/text()').get(),
                'author-link': topic_body.xpath('.//span[@class="creator"]//a').attrib['href'],
                'comment-time': topic_body.xpath('.//span[@class="crawler-post-infos"]/time/text()').get().strip(),
                'comment-timestamp': topic_body.xpath('.//span[@class="crawler-post-infos"]/meta/text()').get(),
                'comment-position': topic_body.xpath('.//span[@class="crawler-post-infos"]/span[@class="position"]/text()').get(),
                'comment-body': topic_body.xpath('.//div[@class="post"]//text()').getall(),
                'comment-likes': topic_body.xpath('.//span[@class="post-likes"]/text()').get()
            })

            # CommentsItem model will hold data and push to db through scrapy pipeline
            item = CommentsItem()
            item["topic"] = topic_url
            item["author"] = topic_body.xpath('.//span[@class="creator"]//span[@itemprop="name"]/text()').get()
            item["author_link"] = topic_body.xpath('.//span[@class="creator"]//a').attrib['href']
            item["comment_time"] = topic_body.xpath('.//span[@class="crawler-post-infos"]/time/text()').get().strip()
            item["comment_body"] =topic_body.xpath('.//div[@class="post"]//text()').getall()
            item["comment_likes"] = topic_body.xpath('.//span[@class="post-likes"]/text()').get()
            yield item

