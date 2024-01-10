import xmltodict
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
import sqlite3
import json
import random
# Extra Fucntions
# Prepare SQLite
# Creates a connection to a SQLite database file.
conn = sqlite3.connect('data.db')
cursor = conn.cursor()  # Creates a cursor object.

# Creating table named 'pages' to store text and source url.
cursor.execute('''
CREATE TABLE IF NOT EXISTS pages (
        text TEXT, 
        source TEXT
    )
                  ''')

conn.commit()  # Commits the transaction.


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0'}

# Proxy
proxies = [
    {
        "http": "http://rahul222rahul222@91.246.195.9:6778"
    },
    {
        "http": "http://rahul222rahul222@103.37.183.170:5869"
    },
    {
        "http": "http://rahul222rahul222@154.85.126.123:5130"
    },
    {
        "http": "http://rahul222rahul222@216.173.88.13:6362"
    },
    {
        "http": "http://rahul222rahul222@64.137.89.78:6151"
    },
    {
        "http": "http://rahul222rahul222@104.239.41.212:6567"
    },
    {
        "http": "http://rahul222rahul222@45.192.143.170:5243"
    },
    {
        "http": "http://rahul222rahul222@104.239.37.121:5773"
    },
    {
        "http": "http://rahul222rahul222@64.137.62.135:5780"
    },
    {
        "http": "http://rahul222rahul222@104.239.23.118:5879"
    },
    {
        "http": "http://rahul222rahul222@161.123.208.240:6484"
    },
    {
        "http": "http://rahul222rahul222@109.196.163.132:6230"
    },
    {
        "http": "http://rahul222rahul222@64.137.75.132:6052"
    },
    {
        "http": "http://rahul222rahul222@93.120.32.13:9197"
    },
    {
        "http": "http://rahul222rahul222@45.192.146.180:6191"
    },
    {
        "http": "http://rahul222rahul222@45.192.140.47:6637"
    },
    {
        "http": "http://rahul222rahul222@107.181.130.236:5857"
    },
    {
        "http": "http://rahul222rahul222@161.123.93.244:5974"
    },
    {
        "http": "http://rahul222rahul222@192.241.112.107:7609"
    },
    {
        "http": "http://rahul222rahul222@104.239.76.176:6835"
    },
    {
        "http": "http://rahul222rahul222@188.74.168.148:5189"
    },
    {
        "http": "http://rahul222rahul222@154.85.125.52:6263"
    },
    {
        "http": "http://rahul222rahul222@103.53.216.62:5146"
    },
    {
        "http": "http://rahul222rahul222@103.99.33.128:6123"
    },
    {
        "http": "http://rahul222rahul222@64.137.65.63:6742"
    },
    {
        "http": "http://rahul222rahul222@64.137.65.88:6767"
    },
    {
        "http": "http://rahul222rahul222@64.137.104.93:5703"
    },
    {
        "http": "http://rahul222rahul222@64.137.88.39:6278"
    },
    {
        "http": "http://rahul222rahul222@104.143.229.197:6125"
    },
    {
        "http": "http://rahul222rahul222@109.207.130.12:8019"
    },
    {
        "http": "http://rahul222rahul222@104.239.0.11:5712"
    },
    {
        "http": "http://rahul222rahul222@161.123.65.204:6913"
    },
    {
        "http": "http://rahul222rahul222@104.239.84.176:6211"
    },
    {
        "http": "http://rahul222rahul222@103.75.228.9:6088"
    },
    {
        "http": "http://rahul222rahul222@216.19.205.155:6476"
    },
    {
        "http": "http://rahul222rahul222@103.99.33.232:6227"
    },
    {
        "http": "http://rahul222rahul222@43.245.116.157:6672"
    },
    {
        "http": "http://rahul222rahul222@216.173.103.141:6655"
    },
    {
        "http": "http://rahul222rahul222@45.192.155.198:7209"
    },
    {
        "http": "http://rahul222rahul222@103.101.88.222:5946"
    },
    {
        "http": "http://rahul222rahul222@104.239.86.37:5947"
    },
    {
        "http": "http://rahul222rahul222@185.48.55.162:6638"
    },
    {
        "http": "http://rahul222rahul222@216.173.105.208:6065"
    },
    {
        "http": "http://rahul222rahul222@103.101.88.197:5921"
    },
    {
        "http": "http://rahul222rahul222@103.101.90.119:6384"
    },
    {
        "http": "http://rahul222rahul222@207.244.217.10:6557"
    },
    {
        "http": "http://rahul222rahul222@103.101.90.159:6424"
    },
    {
        "http": "http://rahul222rahul222@119.42.36.134:6034"
    },
    {
        "http": "http://rahul222rahul222@45.249.106.66:5763"
    },
    {
        "http": "http://rahul222rahul222@207.244.217.81:6628"
    }
]


def extract_text_from(url):
    selected_proxy = random.choice(proxies)
    html = requests.get(url, headers=headers, proxies=selected_proxy).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

   # print(text)

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)


selected_proxy = random.choice(proxies)
r = requests.get("https://peterattiamd.com/post-sitemap.xml",
                 headers=headers, proxies=selected_proxy)
xml = r.text

# print(xml)

raw = xmltodict.parse(xml)

pages = []
for info in raw['urlset']['url']:
    # info example: {'loc': 'https://www.paepper.com/...', 'lastmod': '2021-12-28'}
    url = info['loc']
    if 'https://peterattiamd.com' in url:
        pages.append({'text': extract_text_from(url), 'source': url})
        print(f"Extracted {url}")
        # Save to SQLite
        cursor.execute("INSERT INTO pages VALUES (?,?)",
                       (extract_text_from(url), url))

# save pages to json
with open('pages.json', 'w') as f:
    json.dump(pages, f)

conn.commit()
conn.close()


# text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
# docs, metadatas = [], []
# for page in pages:
#     splits = text_splitter.split_text(page['text'])
#     docs.extend(splits)
#     metadatas.extend([{"source": page['source']}] * len(splits))
#     print(f"Split {page['source']} into {len(splits)} chunks")
