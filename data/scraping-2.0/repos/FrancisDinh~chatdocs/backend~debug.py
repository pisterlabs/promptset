import os
from lib import multithreadCrawler, elastic
import openai
from dotenv import load_dotenv



load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# es = elastic.Elastic(index_name='agent_vector_search');
# print(es.search(agent_id='1232', query='How to create a journey?'));

cc = multithreadCrawler.MultiThreadedCrawler("https://docs.antsomi.com/")
cc.run_web_crawler()
cc.info()