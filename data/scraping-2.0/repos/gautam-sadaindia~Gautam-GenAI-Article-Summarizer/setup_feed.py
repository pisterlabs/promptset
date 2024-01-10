import feedparser
from newspaper import Article
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def setup_feed(rss_feed_url, count):
    feed = feedparser.parse(rss_feed_url).entries[:count]
    article_list = []

    for article in feed:
        article_list.append({"title": article.title, "link": article.link})
    return article_list

def Content(url, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    article = Article(url)
    article.download()
    article.parse()
    texts = text_splitter.split_text(article.text)
    docs = [Document(page_content=i) for i in texts[:3]]
    return docs