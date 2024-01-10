from langchain.document_loaders import RecursiveUrlLoader
import pandas as pd
from urllib.parse import urlparse
from tqdm import tqdm
import os

class url_crawl(RecursiveUrlLoader):
    def __init__(self, base_url, depth):
        super().__init__(url=base_url, max_depth=depth)
        self.base_url = base_url
        self.max_depth = depth

    def get_child_urls(self):
        # Initialize a set to store visited URLs
        visited = set()
        
        # Initialize a list to store the collected URLs
        self.child_urls = []

        # Call the _get_child_links_recursive method to start crawling
        for document in tqdm(self._get_child_links_recursive(self.base_url, visited)):
            self.child_urls.append(document.metadata['source'])  

        return self.child_urls

    def filter_urls(self):
        """ Filter out URLs containing a question mark
        because these urls are not useful for our purpose
        such urls mostly contain search results, css files, etc.
        two things are done here:
        i. filter out urls containing a question mark
        ii. sort the urls in alphabetical order"""

        filtered_urls = (url for url in self.child_urls if '?' not in url)
        #sorting URLS in alphabetical order
        self.sorted_urls = sorted(filtered_urls) 
        return self.sorted_urls        
    
    def process_urls(self):
        """"there are some urls especially in bulletin.iit.edu that
        have duplicate content. One in html form and other in pdf form.
        Here we are doing 2 things mainly:
        1. remove the pdf urls with duplicate content
        2. remove the duplicate urls that result after the first step"""

        # performing step 1
        processed_urls_1 = (
            url.rsplit('/', 1)[0] if url.endswith('.pdf') and 
            urlparse(url).path.split('/')[-1].replace('.pdf', '') == urlparse(url).path.split('/')[-2] 
            else url 
            for url in self.sorted_urls
        )

        # performing step 2
        self.processed_urls_2 = set(url.rstrip('/') for url in processed_urls_1)
        
        return self
        
    def store_urls(self):

        # export to csv
        pd.DataFrame(self.processed_urls_3, columns=['urls']).to_csv('urls_iit_edu.csv')


class MultiCrawler:
    def __init__(self, urls_with_depths):
        # Remove duplicates based on the base URL
        base_urls = {}
        for url, depth in urls_with_depths:
            base_url = urlparse(url).scheme + "://" + urlparse(url).netloc
            if base_url not in base_urls:
                base_urls[base_url] = depth
        self.urls_with_depths = list(base_urls.items())
        self.all_urls = []

    def crawl(self):
        for url, depth in self.urls_with_depths:
            crawler = url_crawl(base_url=url, depth=depth)
            crawler.get_child_urls()
            crawler.filter_urls()
            crawler.process_urls()
            # Assuming process_urls() returns a list of processed URLs
            self.all_urls.extend(crawler.process_urls())

    def get_all_urls(self):
        return self.all_urls

if __name__ == '__main__':
    crawler = url_crawl(base_url='https://www.iit.edu/', depth=3)
    crawler.get_child_urls()
    crawler.filter_urls()
    crawler.process_urls()
    crawler.store_urls()


