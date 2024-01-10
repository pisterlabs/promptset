
import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import openai
import numpy as np
import glob
import logging
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import matplotlib.pyplot as plt

# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])


class WebCrawler():
    full_url : str

    HTTP_URL_PATTERN = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Create a list to store the text files
    texts=[]

    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    max_tokens = 500

    shortened = []

    # Function to get the hyperlinks from a URL
    def get_hyperlinks(self, url):
        
        # Try to open the URL and read the HTML
        try:
            # Open the URL and read the HTML
            with urllib.request.urlopen(url) as response:

                # If the response is not HTML, return an empty list
                if not response.info().get('Content-Type').startswith("text/html"):
                    return []
                
                # Decode the HTML
                html = response.read().decode('utf-8')
        except Exception as e:
            logging.info(e)
            return []

        # Create the HTML Parser and then Parse the HTML to get hyperlinks
        parser = HyperlinkParser()
        parser.feed(html)

        return parser.hyperlinks

    # Function to get the hyperlinks from a URL that are within the same domain
    def get_domain_hyperlinks(self, local_domain, local_path, url):
        clean_links = []
        for link in set(self.get_hyperlinks(url)):
            clean_link = None

            # If the link is a URL, check if it is within the same domain
            if re.search(self.HTTP_URL_PATTERN, link):
                # Parse the URL and check if the domain is the same
                url_obj = urlparse(link)
                if url_obj.netloc == local_domain:
                    clean_link = url_obj.scheme + "://" + url_obj.netloc + url_obj.path

            # If the link is not a URL, check if it is a relative link
            else:
                if link.startswith("/"):
                    link = link[1:]
                elif (
                    link.startswith("#")
                    or link.startswith("mailto:")
                    or link.startswith("tel:")
                ):
                    continue
                # Remove fragment from relative link
                link = link.split('#')[0]
                clean_link = local_path + "/" + link
            if clean_link is not None:
                if clean_link.endswith("/"):
                    clean_link = clean_link[:-1]
                clean_links.append(clean_link)

        # Return the list of hyperlinks that are within the same domain
        logging.info(list(set(clean_links)))
        return list(set(clean_links))

    def crawl(self, url, local_domain, local_path):

        # Create a queue to store the URLs to crawl
        queue = deque([url])

        # Create a set to store the URLs that have already been seen (no duplicates)
        seen = set([url])

        # Create a directory to store the text files
        if not os.path.exists("text/"):
                os.mkdir("text/")

        if not os.path.exists("text/"+local_domain+"/"):
                os.mkdir("text/" + local_domain + "/")

        # Create a directory to store the csv files
        if not os.path.exists("processed"):
                os.mkdir("processed")
        # Get the hyperlinks from the URL and add them to the queue
        for link in self.get_domain_hyperlinks(local_domain, local_path, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)
        # While the queue is not empty, continue crawling
        while queue:

            # Get the next URL from the queue
            url = queue.pop()
            logging.info(url) # for debugging and to see the progress
            parsed = urlparse(url)
            path_parts = parsed.path.split('/')
            filename = path_parts[-1] if path_parts[-1] else path_parts[-2] # handle trailing slash

            # Save text from the url to a <url>.txt file
            with open('text/'+local_domain+'/' + filename + ".txt", "w", encoding="UTF-8") as f:

                # Get the text from the URL using BeautifulSoup
                soup = BeautifulSoup(requests.get(url).text, "html.parser")

                 # Find the main content div
                main_content = soup.find('div', {'class': 'body', 'role': 'main'})

                if main_content is None:

                    # The main content appears to be in <div> with class 'ltx_page_content'
                    main_content = soup.find('div', class_='ltx_page_content')

                if main_content is None:
                    logging.info("Unable to find main content in " + url)
                else:
                    # Get the text but remove the tags
                    text = main_content.get_text()

                    # If the crawler gets to a page that requires JavaScript, it will stop the crawl
                    if ("You need to enable JavaScript to run this app." in text):
                        logging.info("Unable to parse page " + url + " due to JavaScript being required")
                    
                    # Otherwise, write the text to the file in the text directory
                    f.write(text)



