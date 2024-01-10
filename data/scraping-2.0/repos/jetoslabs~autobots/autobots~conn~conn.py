# from functools import lru_cache
#
# from autobots.conn.aws.s3 import S3
# from autobots.conn.duckduckgo.duckduckgo import DuckDuckGo
# from autobots.conn.openai.openai import OpenAI
# from autobots.conn.pinecone.pinecone import Pinecone
# from autobots.conn.selenium.selenium import Selenium
# from autobots.conn.stability.stability import Stability
# from autobots.conn.unsplash.unsplash import Unsplash
# from autobots.core.settings import Settings, get_settings
#
#
# class Conn:
#
#     def __init__(self, settings: Settings):
#         self.open_ai = OpenAI()
#         self.selenium = Selenium()
#         self.stability = Stability()
#         self.unsplash = Unsplash()
#         self.duckduckgo = DuckDuckGo()
#         self.s3 = S3()
#         self.pinecone = Pinecone()
#
#
# @lru_cache
# def get_conn(settings: Settings = get_settings()) -> Conn:
#     return Conn(settings)
