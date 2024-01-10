import os
import openai
import json
import re

from googleapiclient.discovery import build
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

load_dotenv()
apiKey = os.getenv('GOOGLE_API_KEY')
seId = os.getenv('GOOGLE_CSE_ID')
openai.api_key = os.getenv('GPT_KEY')

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res['items']

def query_google(q):
    return google_search(q, apiKey, seId, num=3)

async def gpt_summarize(q):
    text = re.sub(r'\s+', ' ', get_page_text(q))
    print(text)
    link = q['link']
    result = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Create a 2 sentence summary of a website's content using the given text from the website alongside the URL: " + link + " - " + text}])
    return result

def get_page_text(pageObject):
    req = Request(pageObject['link'],headers={'User-Agent': 'Mozilla/5.0'})
    html_page = urlopen(req).read()

    soup = BeautifulSoup(html_page, 'html.parser')
    text = soup.getText()
    soupLen = len(text)
    if soupLen > 3000:
        text = text[:3000]
    return text


#print(json.dumps(query_google("honda civic si"), indent=2))