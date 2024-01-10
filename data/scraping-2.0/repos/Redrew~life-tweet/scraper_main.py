from browser_history import get_history
from browser_history import get_bookmarks
from browser_history.browsers import Chrome
import requests
import html2text
import pickle
import datetime

from openai_api import get_chat_gpt_output
import openai
import pytz


htext_obj = html2text.HTML2Text()
htext_obj.ignore_links = True
htext_obj.ignore_images = True
htext_obj.ignore_tables = True
htext_obj.bypass_tables = True

def get_all_history(browser_type='all', restrict_bookmarks=False):
    if browser_type == 'all':
        outputs = get_history()
    elif browser_type == "Chrome":
        outputs = Chrome().fetch_history()
    else:
        return [
            (0, "https://www.nytimes.com/live/2023/10/14/world/israel-news-hamas-war-gaza", "Israel"),
            (1, "https://www.forbes.com/sites/tomsanderson/2023/10/13/fc-barcelona-captain-sergi-roberto-agrees-to-leave-club-reports/", "Barcelona Soccer"),
            (2, "https://en.wikipedia.org/wiki/Latent_space", "Latent space")
        ]
    user_data = outputs.histories
    print(f'Read {len(user_data)} links')
    return user_data  # [datetime.datetime, url, title, folder]


def get_recent_history(time_limit, browser_type='all', restrict_bookmarks=False):
    current_dateTime = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)
    start_dateTime = current_dateTime - datetime.timedelta(seconds=time_limit)
    
    if browser_type == 'all':
        outputs = get_history()
    elif browser_type == "Chrome":
        outputs = Chrome().fetch_history()
    else:
        return [
            (current_dateTime, "https://www.nytimes.com/live/2023/10/14/world/israel-news-hamas-war-gaza", "Israel"),
            (current_dateTime, "https://www.forbes.com/sites/tomsanderson/2023/10/13/fc-barcelona-captain-sergi-roberto-agrees-to-leave-club-reports/", "Barcelona Soccer"),
            (current_dateTime, "https://en.wikipedia.org/wiki/Latent_space", "Latent space")
        ]
    user_data = outputs.histories
    print(f'Read {len(user_data)} links')
    user_data = list(filter(lambda x: x[0] >= start_dateTime, user_data))
    print(f'Read {len(user_data)} filtered links')
    return user_data  # [datetime.datetime, url, title, folder]

def _clip_token_limit(text, limit=2000):
    return " ".join(text.split(" ")[:limit])

def get_summary_from_html(html: str) -> str:
    # input is html text with tags, output is cleaned [text] segments
    scraped_text = htext_obj.handle(html)
    scraped_text = _clip_token_limit(scraped_text, limit=2000)
    summary = ""
    try:
        summary = get_chat_gpt_output(f"A user visited a website. Given the following HTML code, give a brief summary of the text. {scraped_text}")
    except openai.error.InvalidRequestError as e:
        pass
    return summary


def scrape_websites(history, ignore_google=True, verbose=False):
    raw_html = []
    for datetime, url, title in history:
        if ignore_google and title.endswith('Google Search'):
            continue
        if verbose:
            print(f'Title: {title}')
        try:
            f = requests.get(url)
            raw_html.append(f.text)
        except requests.exceptions.InvalidSchema as e:
            print(f"Couldn't read website {url}")
    return raw_html


def get_data_from_browser(use_recent = False, ignore_google=True, time_limit=600, browser_type='all'):
    if use_recent:
        history = get_recent_history(time_limit=time_limit, browser_type=browser_type)
    else:
        history = get_all_history(browser_type=browser_type)
    print(history)
    raw_html = scrape_websites(history, ignore_google=ignore_google)
    print('parsing texts scraped from html')
    website_texts = [get_summary_from_html(html) for html in raw_html]
    print(website_texts)
    website_texts = [text for text in website_texts if 'html' not in text.lower()]
    return website_texts

if __name__ == '__main__':
    website_texts = get_data_from_browser(use_recent=True, time_limit=240, ignore_google=True, browser_type='Chrome')
    with open('browser_clean.pkl', 'wb') as handle:
        pickle.dump(website_texts, handle)