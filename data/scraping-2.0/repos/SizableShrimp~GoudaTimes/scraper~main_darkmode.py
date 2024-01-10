import openai
import os
from datetime import datetime

from article import ArticleNPK
from articles_db import ArticlesDb

from actually_hash import actually_hash

def generate_dark_article_title():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "user",
            "content": "Come up with the darkest cheese-related news headline you can. They must be cheese-focused and dramatic. ",
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    title = response["choices"][0]["message"]["content"]
    if title.startswith("'") and title.endswith("'"):
        title = title[1:-1]
    if title.startswith('"') and title.endswith('"'):
        title = title[1:-1]
    return title

def generate_dark_article_content(title: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """
                I will give the title of a news article.
                Write the article.
                It must be about cheese, very dark, and excessively dramatic.
                No matter what, respond with only the article content.
                Do not add any other information or formatting.
                """.strip(),
            },
            {
            "role": "user",
            "content": title,
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response["choices"][0]["message"]["content"]

def generate_dark_article() -> ArticleNPK:
    print("[t", end="", flush=True)
    title = generate_dark_article_title()
    print("][c", end="", flush=True)
    content = generate_dark_article_content(title)
    print("]", end="", flush=True)
    return ArticleNPK(
        title,
        "<< not available >>",
        datetime.now().isoformat()[:10],
        "By an anonymous depressed cheesemonger",
        None,
        None,
        "darkmode._",
        "{}",
        f"https://chat.openai.com?_darkmodearticletitle={actually_hash(title)}",
        content,
        skip_cheesify=True,
    )

if __name__ == "__main__":
    print("~> Setting up database connection...", end="", flush=True)
    db = ArticlesDb("already_processed.pkl")
    print("done.")
    n = 25
    for i in range(n):
        print(f"~> Generating article {i+1}/{n}...", end="", flush=True)
        article = generate_dark_article()
        print("done.")
        print(f"Generated article \"{article.title}\".")
        db.add_article(article, f"{i+1}/{n}")
