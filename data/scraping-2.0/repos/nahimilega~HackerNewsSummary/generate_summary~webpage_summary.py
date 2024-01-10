import requests
import openai

from bs4 import BeautifulSoup
import app_config
import json
from urllib.parse import urlparse

openai.api_key = app_config.openai_Key

blacklist_domain = [
    "twitter.com",
    "youtube.com",
    "vimeo.com",
    "instagram.com",
    "x.com",
    "reddit.com",
    "latimes.com",
]


context = """You are an expert summarizer working for a Hacker News. You are given the title and content of an article and you need to generate a dehyped title and summary for the article.
"""

prompt = """Here is an article posted on Hacker News.
### Title: {title}
### URL: {url}

You are given the scraped content of URL.
{content}



Make a dehyped title for the article that capture gist of the article in a single, dense sentence. Keep is different and more informative from the title of the article. Do not repeat the title of the article in the dehyped title.

After that please provide a TLDR of the content of the webpage. Focus on the main topics and key points discussed. 
The goal is to provide enough information to gauge the article's relevance without being too lengthy. Avoiding Boilerplate Language. Do now repeat the title of the article in the summary.

Keep it in pointer and very concise and stick to the facts and and key themes discussed.

This would be posted on hacker news. Keep the title and summary according to what people would like to read on hacker news.
"""

functions = [
    {
        "name": "dehyped_title_and_summary",
        "description": "For the given title and content of the HackerNews post, generate a dehyped title and summary for the article.",
        "parameters": {
            "type": "object",
            "properties": {
                "dehyped_title": {
                    "type": "string",
                    "description": "Generate a Dehyped title for the HackerNews post that capture gist of the article in a single, dense sentence.",
                },
                "article_summary": {
                    "type": "string",
                    "description": "Summary of the article. The goal of this summary is to help readers decide whether to read the full article. Make sure this would be posted on hacker news. Keep it in pointers and super concise.",
                },
            },
            "required": [
                "dehyped_title",
                "article_summary",
            ],
        },
    }
]


def get_prompt(story_data, content: str) -> str:
    content = content[:20000]

    return prompt.format(
        title=story_data.title,
        url=story_data.url,
        content=content,
    )


def make_summary(story_data, content: str, retry_count: int = 0) -> dict:
    userPrompt = get_prompt(story_data, content)

    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": userPrompt},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        functions=functions,
        temperature=0 + retry_count * 0.1,
        max_tokens=1000,
        presence_penalty=0,
        frequency_penalty=0,
        function_call={
            "name": "dehyped_title_and_summary"
        },  # auto is default, but we'll be explicit
    )

    response_json = json.loads(
        response["choices"][0]["message"]["function_call"]["arguments"]
    )
    if "dehyped_title" not in response_json or "article_summary" not in response_json:
        raise Exception("Failed to generate summary")
    return response_json


def if_url_is_blacklisted(url: str) -> bool:
    domain = urlparse(url).netloc
    for blacklist_domain_name in blacklist_domain:
        if domain in blacklist_domain_name:
            return True
    return False


def get_webpage_text(url: str) -> str:
    if if_url_is_blacklisted(url):
        raise Exception("URL is blacklisted")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Referer": "https://www.google.com/",
    }

    response = requests.get(url, timeout=10, headers=headers)

    if response.status_code == 200:
        # Using BeautifulSoup to parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Extracting the body of the HTML
        body = soup.body.get_text(separator=" ", strip=True)

    else:
        ## Throw an error
        print(f"Failed to retrieve content, status code: {response.status_code}")
        raise Exception(
            f"Failed to retrieve content, status code: {response.status_code}"
        )
    return body


def generate_webpage_summary(story_data) -> tuple[str, str]:
    retry_count = 0
    while retry_count < 4:
        retry_count += 1
        try:
            content = get_webpage_text(story_data.url)
            summary = make_summary(story_data, content, retry_count)
            return summary["dehyped_title"], summary["article_summary"]
        except Exception as e:
            print(f"Failed to generate summary for {story_data.url}: {e}")

    return "", ""
