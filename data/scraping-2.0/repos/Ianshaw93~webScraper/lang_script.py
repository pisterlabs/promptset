# # from langchain.document_loaders import UnstructuredURLLoader
# import os
# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 
# import requests
# from bs4 import BeautifulSoup

# # get urls
# BASE_URL = "https://kneesovertoesguy.medium.com/"

# def get_article_links():
#     response = requests.get(BASE_URL)
#     soup = BeautifulSoup(response.content, 'html.parser')
    
#     # Extract all <article> tags
#     articles = soup.find_all('article')
    
#     # Extract the href attribute from the nested <a> tags within each <article>
#     article_links = [article.find('a', href=True)['href'].split('?')[0] for article in articles if article.find('a', href=True)]
    
#     return article_links

# article_links = [BASE_URL + f for f in get_article_links()]
# loaders = UnstructuredURLLoader(urls=article_links)
# data = loaders.load()
