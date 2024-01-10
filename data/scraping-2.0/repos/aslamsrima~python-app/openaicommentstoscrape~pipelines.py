# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import psycopg2

from openaicommentstoscrape.items import CommentsItem, TopicItem


class OpenaicommentstoscrapePipeline:
    # when spider is opened connect db and store handle and cursor
    def open_spider(self, spider):
        hostname = '192.168.1.12'
        username = 'limoo'
        password = 'limoo'
        database = 'topics'
        self.connection = psycopg2.connect(
            host=hostname, user=username, password=password, dbname=database)
        self.cur = self.connection.cursor()

        # create tables if not exists
        self.cur.execute("CREATE TABLE IF NOT EXISTS topics(id SERIAL PRIMARY KEY, topic VARCHAR(255), replies VARCHAR(255), view VARCHAR(255), activity VARCHAR(255), topic_url VARCHAR(255))")
        self.cur.execute("CREATE TABLE IF NOT EXISTS comments(topic VARCHAR(255), author VARCHAR(255), author_link VARCHAR(255), comment_time VARCHAR(255), comment_body TEXT, comment_likes VARCHAR(255))")

    # when spider is closed, close db cursor and connection
    def close_spider(self, spider):
        self.cur.close()
        self.connection.close()

    def process_item(self, item, spider):
        # Check if item is instance of TopicItem, insert in topic table
        if isinstance(item, TopicItem):
            self.cur.execute(
                'INSERT INTO topics(topic, replies, view, activity, topic_url) VALUES(%s, %s, %s, %s, %s)', (item['topic_name'], item['replies'], item['views'], item['activity'], item['topic_url']))

        # Check if item is instance of CommentsItem, insert in comments table
        if isinstance(item , CommentsItem):
            self.cur.execute('INSERT INTO comments(topic, author, author_link, comment_time, comment_body, comment_likes) VALUES(%s, %s, %s, %s, %s, %s)', (item["topic"], item["author"], item["author_link"], item["comment_time"], item["comment_body"], item["comment_likes"]))

        self.connection.commit()
        return item

