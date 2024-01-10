import csv
import yaml
import json
import openai
import tiktoken
import asyncio
import aiohttp
from glob import glob
from pathlib import Path
from requests.compat import urljoin
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Optional
from html2text import HTML2Text

# read config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# setup config
openai.api_key = config['openai_api_key']
urls = config['scrape_urls']
max_depth = config['max_depth']

enc = tiktoken.get_encoding('cl100k_base')

h = HTML2Text()
h.ignore_links = True
h.ignore_images = True

class Scraper:
    def __init__(self, session: aiohttp.ClientSession, base_url: str, max_depth: int):
        self.session = session
        self.base_url = base_url
        self.base_netloc = urlparse(self.base_url).netloc
        self.max_depth = max_depth
        self.visited = set()
        self.data = {}

    async def get_data_from_url(self, url: str) -> Optional[tuple[str, list[str]]]:
        try:
            async with self.session.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0',
            }) as response:
                data = await response.text()
                soup = BeautifulSoup(data, 'html.parser')
                content = h.handle(data)

                # Extracting all anchor tags
                anchors = soup.find_all('a', href=True)
                
                # Extracting links and making them absolute links if they are relative
                links = [urljoin(self.base_url, anchor['href']) for anchor in anchors]
                
                return content, links
        except:
            return None

    # classic dfs
    async def scrape(self, url: str, depth: int = 0):
        if (url in self.visited) or (urlparse(url).netloc != self.base_netloc) or (depth > self.max_depth):
            return
        self.visited.add(url)

        # extract links from the current url
        print(f'Depth: {depth} | URL: {url}')
        data = await self.get_data_from_url(url)

        if not data:
            print('No data found')
            return

        content, links = data
        self.data[url] = content

        for link in links:
            await self.scrape(link, depth + 1)

    async def begin(self):
        await self.scrape(self.base_url, 0)

        # save data
        with open(f'./scraper_db/{self.base_netloc}.json', 'w') as f:
            json.dump(self.data, f)


async def scrape_urls(urls):
    async with aiohttp.ClientSession() as session:
        corrs = []
        for url in urls:
            scraper = Scraper(session, url, max_depth)
            corrs.append(scraper.begin())

        await asyncio.gather(*corrs)

def truncate(x):
    return enc.decode(enc.encode(x)[:16000])

def parse(file_path):
    with open(file_path, 'r') as f:
        site_data = json.load(f)
        data = ''
        for k, v in site_data.items():
            if 'privacy' in k or 'terms' in k: continue
            data += '-- ' + k + '\n'
            data += v + '\n'
    
    messages = [
      { "role": "system", "content": "You are an expert data analyzer. You will be provided with web scraped data in markdown format of a startup website. The scraped data provided to you will follow a specific format. Since the data is scraped from multiple pages of the website you will be provided with both the link and the content of that link. The link will begin with '--' and following the link will be the website's content. You should extract information about the target customer segment of the product only. Only use the context that provide you with relevent information and ignore irrelevant context."},
      {"role": "user", "content": f"Here is the web scraped data:\n{truncate(data)}"},
    ]

    response = openai.ChatCompletion.create(
      model='gpt-3.5-turbo-16k',
      messages=messages,
    )

    website = file_path.split('/')[-1].replace('.json', '')
    print(f'Analysis({website}): {response.choices[0].message.content}\n') # type: ignore

    with open('./scraper_db/openai_analysis.raw', 'a') as f:
        s = f"-------------------- {website} --------------------\n"
        s += response.choices[0].message.content # type: ignore
        s += '\n\n\n'
        f.write(s)
    with open('./scraper_db/openai_analysis.csv', 'a') as f:
        w = csv.writer(f)
        w.writerow([website, response.choices[0].message.content]) # type: ignore

def generate_analysis():
    fs = glob('./scraper_db/*.json')
    with open('./scraper_db/openai_analysis.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(["Website", "Description"]) # type: ignore
    for f in fs:
        parse(f)

async def main():
    Path('./scraper_db').mkdir(parents=True, exist_ok=True)
    await scrape_urls(urls)
    print('\nDone scrapping')
    print('\nAnalyzing documents now...')
    generate_analysis()
    print('\nAnalysis saved to ./scraper_db/openai_analysis.raw and ./scraper_db/openai_analysis.csv')


if __name__ == "__main__":
    asyncio.run(main())
