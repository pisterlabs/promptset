import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import json
import os
from dotenv import load_dotenv
load_dotenv()


def extract_links_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]


def extract_main_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        soup = BeautifulSoup(response.content, 'lxml')
        # Here you can add logic to extract the main content
        # For example, if the main content is in a specific tag
        main_content = soup.find('main')  # This is just an example
        return main_content.text if main_content else None
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


def extract_headline(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Attempt to find the headline in an <h1> tag or <title> tag
        headline = soup.find('h1') or soup.find('title')
        return headline.text.strip() if headline else None
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


def makeSoup(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup


def getHrefs(url):
    soup = makeSoup(url)
    # find all links on page 
    hrefs = [link.get('href') for link in soup.find_all('a')]
    # remove duplicates
    hrefs = list(dict.fromkeys(hrefs))
    # remove empty links
    hrefs = [href for href in hrefs if href is not None]
    # remove links that start with #
    hrefs = [href for href in hrefs if not href.startswith('#')]
    # remove links that only have one / 
    hrefs = [href for href in hrefs if href.count('/') > 2 ]
    # remove links that don't start with base url or /
    if 'ycombinator' not in url:
        hrefs = [href for href in hrefs if href.startswith('http') or href.startswith('/')]

    print("URLs saved: ", len(hrefs))

    return hrefs


def scrape_urls(urls, output_file):
    with open(output_file, 'w') as file:
        for url in urls:
            print(url)
            hrefs = getHrefs(url)
            for href in hrefs:
                if not href.startswith('http'):
                    file.write(url + href + '\n')
                else:
                    file.write(href + '\n')
    file.close()


def get_data(urls):
    scrape_urls(urls, 'hrefs.txt')
    file_path = 'hrefs.txt'  # Replace with your file path
    urls = extract_links_from_file(file_path)

    articles = []
    for url in urls:
        content = extract_main_content(url)
        headline = extract_headline(url)

        article = {
            "url": url,
            "title": headline,
            "content": content
        }
        articles.append(article)

    with open('articles.json', 'w') as file:
        json.dump(articles, file)