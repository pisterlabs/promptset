from abc import ABC, abstractmethod
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from selenium.webdriver.support import expected_conditions as EC
from typing import List, Tuple
from time import time

import asyncio
import aiohttp
from bs4 import BeautifulSoup

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

from typing import Any, Dict, List, Union



class SearchCrawler(ABC):
    """
    Abstract Base Class for search crawlers
    """
    def __init__(self):
        self.results = None

    @abstractmethod
    def search(self):
        pass


    def get_results(self, query: str, max_results:int = 3) -> List[Tuple[str, str, str]]:
        if self.results is None:
            self.search(query, max_results)
        return self.results

class DuckDuckGoNews(SearchCrawler):
    """
    DuckDuckGo News search crawler
    """
    def __init__(self, driver):
        super().__init__()
        self.driver = driver

    def search(self, query, max_results=3):
        print(query)
        t = time()
        url = f"https://news.duckduckgo.com/?q={query.replace(' ', '+')}&iar=news&ia=news"
        # navigate to the DuckDuckGo search page
        self.driver.get(url)
        headings = WebDriverWait(self.driver, 10).until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR, '.result__a')))
        descs = WebDriverWait(self.driver, 10).until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR, '.result__snippet')))
        
        #min of result or max_results
        max_results = min(max_results, len(headings))
        headings = headings[:max_results]
        descs = descs[:max_results]

        links = [l.get_attribute('href') for l in headings]
        headings = [l.text for l in headings]
        descs = [l.text for l in descs]

        self.results = [links, headings, descs]

class Page:
    def __init__(self, url: str, max_tokens = 1000):
        """
        Initialize a Page object with a given URL.

        :param url: The URL of the webpage to fetch and parse.
        """
        self.json_dict = None
        self.text = ''
        self.heading = None
        self.url = url
        self.links = []
        self.metadata = ''
        self.headings = []
        self.max_tokens = max_tokens
        # self.url = url
        # self.description = None
        # self.keywords = []
    
    def parse_json(self):
        first_heading = True

        for d in self.json_dict:
            if d.get("text") == "Metadata":
                for c in d.get('content', []):
                    self.metadata += c.get("text") + '\n' 
            elif d.get("tag_name").startswith("h"):
                if first_heading:
                    self.heading = d.get("text")
                    first_heading = False
    
                self.headings.append(d.get("text"))
                self.text += d.get("text") + "\n"
            for c in d.get("content", []):
                if c.get("tag_name").startswith("h"):
                    self.headings.append(c.get("text"))
                    self.text += c.get("text") + "\n"
                elif c.get("tag_name") == "ul":
                    self.text += c.get("text").replace("\n", "") + "\n"
                    for item in c.get("list_items", []):
                        self.text += item + "\n"
                elif c.get("tag_name") == "p":
                    self.text += c.get("text") + "\n"
            if len(self.text.split()) > self.max_tokens:
                break

    async def fetch_url(self, session, url):
        """
        Asynchronously fetches the content of a web page.

        :param session: The aiohttp ClientSession object to use for the request.
        :param url: The URL of the page to be fetched.
        :return: The textual content of the page.
        """
        async with session.get(url) as response:
            return await response.text()


    async def parse_content(self, html_content, ignore_header_footer = False):
        """
        Asynchronously parses the HTML content of the page and populates its attributes.

        :param ignore_header_footer: A boolean indicating whether to ignore the header and footer sections of the page.
        :return: The current instance of the Page class.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        async with aiohttp.ClientSession(headers = headers) as session:
          try:
            # send request to the given URL
            # response = requests.get(self.url, headers = headers)
            # parse the HTML content using BeautifulSoup

            # find the main content area of the webpage (ignoring header and footer)
            main_content = soup.find('body')
            if ignore_header_footer:
                header = main_content.find('header')
                footer = main_content.find('footer')
                if header:
                    header.extract()
                if footer:
                    footer.extract()
            # find the main content section of the article
            article = main_content.find('article')

            if not article or len(article) ==0:
              article = main_content

            
            # create an empty array to store JSON objects
            data = []
            
            # create a dummy header object for text before the first header tag
            dummy_header = {'tag_name': 'h0', 'text': 'Metadata', 'content': []}
            
            # flag to keep track of whether we have encountered the first header tag
            found_first_header = False
            
            # add all text and list elements before the first header tag to the dummy header object
            for tag in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ol', 'ul']):
                text = tag.text.strip().replace(u'\xa0', u' ').replace('\n\n', ' ').replace('"', '\\"').replace(u'\u2026', '...').replace(u'\u2019', "'").replace(u'\u201c', '"').replace(u'\u201d', '"')
                if not found_first_header:
                    # check if the current tag is a header tag
                    if tag.name.startswith('h'):
                        found_first_header = True
                        data.append({'tag_name': tag.name, 'text': text, 'content': []})
                    # if we haven't encountered the first header tag yet,
                    # add the element to the dummy header object
                    else:
                        dummy_header['content'].append({'tag_name': tag.name, 'text': text})  
                else:
                    # create a JSON object for each element
                    obj = {'tag_name': tag.name, 'text': text}
                    
                    # if the element is an ordered list or unordered list,
                    # extract the list items and add them to the JSON object
                    if tag.name == 'ol' or tag.name == 'ul':
                        items = []
                        for li in tag.find_all('li'):
                            items.append(li.text.strip())
                        obj['list_items'] = items
                    
                    # add the JSON object to the array
                    if tag.name.startswith('h'):
                        # create a new object for each heading and add it to the array
                        data.append({'tag_name': tag.name, 'text': text, 'content': []})
                    else:
                        # add the object to the last heading object in the array
                        data[-1]['content'].append(obj)
        
            
            # add the dummy header object to the array if there is content
            if dummy_header['content']:
                data = [dummy_header] + data
        

            # assign the array of JSON objects
            self.json_dict = data

            self.parse_json()

            # self.links = parse_links(html_content, self.url)
          except:
            print('Timeout')

        return self


    async def parse(self, ignore_header_footer=False):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        async with aiohttp.ClientSession(headers=headers) as session:
            try:
                html_content = await asyncio.wait_for(self.fetch_url(session, self.url), timeout=2)

                # concurrently parse the HTML content
                await asyncio.gather(self.parse_content(html_content, ignore_header_footer))
            except:
                print('Timeout')

        return self