from bs4 import BeautifulSoup
import requests
import argparse
from langchain.document_loaders import AsyncChromiumLoader, AsyncHtmlLoader
from langchain.document_transformers import BeautifulSoupTransformer
import trafilatura
import pandas as pd
import json



HEADERS = {'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'}


parser = argparse.ArgumentParser(__name__)
parser.add_argument("--url")

args = parser.parse_args()
url = args.url


html = trafilatura.fetch_url(url)
data = trafilatura.extract(html).split('\n')

df = pd.DataFrame.from_records(data)
with open(f"{url.split('/')[-1]}.json", 'w') as js_file:
    json.dump({'content': data}, js_file, ensure_ascii=False)



