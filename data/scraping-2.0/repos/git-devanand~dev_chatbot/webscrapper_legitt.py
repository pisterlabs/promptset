import requests # Required to make HTTP requests
from bs4 import BeautifulSoup # Required to parse HTML
import numpy as np # Required to dedupe sites
from urllib.parse import unquote # Required to unquote URLs
from legitt_sitemap_urls import sitemap_url_extractor   # urls extracted from sitemap
import json


# loop over `links` and keep only the one that have the href starting with "/url?q="
# urls = [
#     'https://legitt.xyz/',
#     'https://legitt.xyz/about-us',
#     'https://legitt.xyz/blog',
#     'https://legitt.xyz/pricing',
#     'https://legitt.xyz/product-tour',
#     'https://legitt.xyz/smart-contract',
# ]

sitemap_url = "https://legitt.xyz/sitemap_index.xml"

urls = sitemap_url_extractor(sitemap_url)
urls = [url[0] for url in urls]

file_name_url = "./datasets/sitemap_urls"
with open(file_name_url, "w") as f:
    f.write(str(urls))

print("Sitemap urls saved to {} file successfully.".format(file_name_url))

print("Total number of urls extraceted: " + str(len(urls)))

print(type(urls))


# Use numpy to dedupe the list of urls after removing anchors

urls = list(np.unique(urls))
urls

from readabilipy import simple_json_from_html_string # Required to parse HTML to pure text
from langchain.schema import Document # Required to create a Document object
# from readabilipy.simple_json import simple_json_from_html_string

unicodedecodeerrored_urls = []

def scrape_and_parse(url: str) -> Document:
    try:
        """Scrape a webpage and parse it into a Document object"""
        req = requests.get(url)
        content = req.content.decode('utf-8')

        # print("###################### Request Text {} ###############################".format(url))
        
        article = simple_json_from_html_string(content, 
                                                content_digests=False,  # (optional, default: False): When set to True, this parameter enables the extraction of content digests from the HTML. Content digests are short summaries or representations of the main content of a web page.
                                                node_indexes=False, # (optional, default: False): When set to True, this parameter includes the node indexes in the JSON output. Node indexes are the positions of HTML elements in the document tree.
                                                use_readability=True,   # (optional, default: False): When set to True, this parameter activates the usage of the Readability algorithm to extract the main content from the HTML. The Readability algorithm attempts to identify and isolate the relevant textual content from the noise and clutter present in a web page.
                                                )
        
        # print("##################### Article Text {} #############################".format(url))
       
        # The following line seems to work with the package versions on my local machine, but not on Google Colab
        if article is not None:
            return Document(page_content=article['plain_text'][0]['text'], metadata={'source': url, 'page_title': article['title']})
        else:
            return None
        
        # The following line works on Google Colab
        # return Document(page_content='\n\n'.join([a['text'] for a in article['plain_text']]), metadata={'source': url, 'page_title': article['title']})


    except UnicodeDecodeError:
        unicodedecodeerrored_urls.append(url)
        # print(f"Appended Error decoding content for URL to list {len(unicodedecodeerrored_urls)}: {url}")
        print(f"{len(unicodedecodeerrored_urls)}. UnicodeDecodeError: Error decoding content for URL at: {url}")
        # return None
        pass


#  remove url values from url which exists in    
# It's possible to optitimize this by using asyncio
documents = [scrape_and_parse(f) for f in urls] # Scrape and parse all the urls


log_err = "./logs/unicodedecodeerrored_urls"

with open(log_err, "w") as log:
    # log.write(str(unicodedecodeerrored_urls))
    log.write(",".join(unicodedecodeerrored_urls))



print("UnicodeDecodeError urls logged to {} file.".format(log_err))

file_name = "./datasets/legitt_documents"
with open(file_name, "w") as f:
    f.write(str(documents))



print("Documents saved to the {} file.".format(file_name))



