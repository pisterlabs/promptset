import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import openai
import time
import datetime as dt
from pathlib import Path
import os
import tqdm

openai.api_key = os.environ.get('OPENAI_SECRET_KEY')

programming_languages = [
    "python",
    "javascript",
    "c++",
    "ansible",
    "bash",
]

programming_methods = [
    "list comprehension",
    "for loop",
    "while loop",
    "if statement",
    "function",
    "class",
    "object",
    "dictionary",
    "list",
    "tuple",
    "set",
]


def get_tech_articles():
    hacker_news_url = f"https://news.ycombinator.com/"
    response = requests.get(hacker_news_url)
    soup = BeautifulSoup(response.text, "html.parser")
    # grab anchor tags with href attribute that doesn't start with "item?id=" and includes "https://" in the href
    links = soup.find_all('a', href=lambda href: href and not href.startswith(
        'item?id=') and 'https://' in href)

    # get the text and href attributes of the links
    articles = []
    for link in tqdm.tqdm(links[0:10]):
        article = {
            'title': link.text,
            'href': link['href'],
        }

        # visit each link and get the biggest body of text from the page
        try:
            response = requests.get(article['href'])
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            article['paragraph'] = text
            articles.append(article)
        except:
            continue

    # run the summaries
    for article in tqdm.tqdm(articles):
        article['paragraph'] = article['paragraph'][:4096]
        prompt = f"Please rewrite the following article as a complete one minute YouTube video. Keep the summary as concise as possible, but make it engaging and interesting, in the style of a VSauce video without ever referring to the narrator. If necessary, provide your own examples. Here is the title: {article['title']}. I'm feeding you the entire webpage's paragraph elements, so ignore any random or unrelated information: {article['paragraph']}"

        generate_summary(
            title=article['title'], prompt=prompt, directory='tech_video_scripts')

        time.sleep(2)

    return articles


def get_programming_examples(language, method):
    # search web.dev, stackoverflow, and freeCodeCamp for examples of the method
    google_url = f"https://www.google.com/search?q=site:web.dev+site:stackoverflow.com+site:freecodecamp.org+{language}+{method}"

    response = requests.get(google_url)
    soup = BeautifulSoup(response.text, "html.parser")
    # grab anchor tags with href attribute that include "https://" in the href
    links = soup.find_all('a', href=lambda href: href and 'https://' in href)

    # visit each link and get the biggest body of text from the page
    for link in links[3:4]:
        try:
            response = requests.get(link['href'])
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            generate_summary(
                title=f"{language} {method}", prompt=text, directory='programming_video_scripts')
        except:
            continue

        time.sleep(2)

    return links


def get_history_articles():
    '''
    Should scrape random subreddits for top posts from the day and run them through the model. Need to use selenium to get the full text of the post and any comments if necessary.

    Returns:
        list: list of dictionaries with the post_title and post_url -> run through the summary function
    '''
    random_history_subreddits = [
        "worldhistory",
        "anthropology",
        "archaeology",
        "ushistory",
        "americanhistory",
        "uscivilwar",
        "mesoamerica",
        "uspresidentialhistory",
        "historyoftheamericas",
        "historyoftexas",
    ]
    history_url = f"https://www.reddit.com/r/{random_history_subreddits[0]}"

    # use selenium to get the full text of the post and any comments if necessary
    driver = webdriver.Firefox()
    driver.get(history_url)
    time.sleep(5)

    # get each post
    posts = driver.find_elements(By.TAG_NAME, "h3")
    # only show the content inside the h3 tag
    posts = [post.text for post in posts]

    for post in posts:
        if post != '':
            prompt = f"I'm going to give you a title of a post from a random history subreddit. Thoroughly research the post and write a complete and concise one minute YouTube video about the post. In the style of a VSauce video without ever referring to the narrator. Imagine you're explaining to someone who has never heard of it before, but never refer to the post as a post. If it's applicable, show the historical context and the cultural and social impact of the post. Here is the title: {post}."
            generate_summary(title=post, prompt=prompt,
                             directory='history_video_scripts')
            time.sleep(2)

    driver.quit()
    print(posts)


def generate_summary(title, prompt, directory):
    # You can use the following keywords to help you: {topic}."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a script writer for a YouTube channel that makes one minute videos about random topics. You are given a title of a post from a random subreddit. You must research the post and write a complete and concise one minute YouTube video about the post. In the style of a VSauce video without ever referring to the narrator. Imagine you're explaining to someone who has never heard of it before, but never refer to the post as a post. If it's applicable, show the historical context and the cultural and social impact of the post."},
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    summary = response.choices[0].message.content
    # write to a txt file with the title and the date as the filename
    project_dir = Path(__file__).resolve().parents[1]
    filename = f"{directory}/{title} {dt.datetime.now().strftime('%Y-%m-%d')}.txt"
    with open(filename, 'w') as f:
        f.write(summary)

    print(f"Summary for {title} written to {filename}")

    return summary


if __name__ == "__main__":
    get_tech_articles()
    # get_programming_examples(programming_languages[0], programming_methods[3])
    # get_history_articles()

