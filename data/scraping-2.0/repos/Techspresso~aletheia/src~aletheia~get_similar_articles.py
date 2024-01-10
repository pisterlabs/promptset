from functools import lru_cache
from langchain.document_loaders import BraveSearchLoader
from aletheia.cache import cache

from aletheia.get_article import get_article_content

api_key = "BSAt2nmuC57jmjrGEY9-JNAyAHTU6Z5"

def get_urls_on_topic(topic, url, count=4):
    docs = search(topic, count=count)
    urls = [doc.metadata["link"] for doc in docs if str(doc.metadata["link"]) != str(url)]
    url_list = urls[:3]
    print("Got urls for topic: " + topic + " : " + str(url_list))
    return url_list

def get_articles_from_urls(urls):
    articles = get_article_content(urls)
    return articles

@cache
def get_articles_on_topic(topic, url, count=3):
    return get_articles_from_urls(get_urls_on_topic(topic, url, count=count))

def search(topic, count=4):
    loader = BraveSearchLoader(
        query=topic, api_key=api_key, search_kwargs={"count": count}
    )
    return loader.load()

if __name__ == "__main__":
    print(get_articles_from_urls(get_urls_on_topic("hamas and israel conflict","https://www.cnbc.com/2023/11/04/israel-hamas-war-live-updates-latest-news-on-gaza-conflict.html")))