import os
import snscrape.modules.twitter as sntwitter
import requests
from bs4 import BeautifulSoup
import openai
import re

# Set up OpenAI API Key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Function to get the latest headlines from The Guardian
def get_latest_headlines(user, num_headlines):
    headlines = []
    for i, tweet in enumerate(sntwitter.TwitterUserScraper(user).get_items()):
        if i >= num_headlines:
            break
        content = re.sub(r"(https?://\S+|#\S+)", "", tweet.rawContent).strip()
        headlines.append((content, tweet.tcooutlinks[0] if tweet.tcooutlinks else None))
    return headlines

# Function to extract the first 1000 characters of paragraph text and a relevant photo URL from a URL
def extract_text_and_photo(url):
    if not url:
        return None, None
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])[:1000]

    picture_tag = soup.find('picture')
    if picture_tag:
        source_tag = picture_tag.find('source')
        if source_tag:
            photo_url = source_tag.get('srcset')
        else:
            photo_url = None
    else:
        photo_url = None

    return text, photo_url

# Function to summarize text in the style of a news report using OpenAI
def summarize_text(text):
    if not text:
        return None
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Please summarize the following text in the style of a news report:\n{text}\n",
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7,
    )

    summary = response.choices[0].text.strip()
    return summary

# Fetch the headlines
num_headlines = 10
guardian_headlines = get_latest_headlines('guardian', num_headlines)

# Process the headlines and save to a file
with open('headlines.md', 'w') as f:
    for i, (headline, url) in enumerate(guardian_headlines):
        text, photo_url = extract_text_and_photo(url)
        summary = summarize_text(text)
        f.write(f"### {i + 1}. {headline}\n\n")
        if summary:
            f.write(f"{summary}\n\n")
            if photo_url:
                f.write(f"![Image]({photo_url})\n\n")
        else:
            f.write("No summary available.\n\n")

print("Headlines and summaries saved to 'headlines.md'")
