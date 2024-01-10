import requests
import xml.etree.ElementTree as ET
import json
from openai import OpenAI
import keys


# Initialize OpenAI client
openai_client = OpenAI(api_key=keys.openai_api_key)  # Replace with your actual API key

def get_hacker_news_rss_feed(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching RSS feed: {e}")
        return None

def summarize_articles(articles_text):
    conversation = [
        {"role": "system", "content": "You are a chatbot designed to provide a summary of multiple news articles."},
        {"role": "user", "content": f"Summarize these articles: {articles_text}"}
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=conversation
    )

    summary = response.choices[0].message.content
    return summary

def parse_and_summarize_rss_feed(data):
    root = ET.fromstring(data)
    all_articles_text = ""

    for item in root.findall('.//item'):
        title = item.find('title').text.strip()
        description = item.find('description').text.strip()
        all_articles_text += f"Title: {title}\nDescription: {description}\n\n"

    return summarize_articles(all_articles_text)

def write_to_file(summary, filename='threat_intel.json'):
    with open(filename, 'w') as file:
        json.dump([{"role": "system", "content": summary}], file, indent=4)

def main():
    url = "https://feeds.feedburner.com/TheHackersNews"
    rss_feed_data = get_hacker_news_rss_feed(url)
    if rss_feed_data:
        summary = parse_and_summarize_rss_feed(rss_feed_data)
        write_to_file(summary)

if __name__ == "__main__":
    main()
