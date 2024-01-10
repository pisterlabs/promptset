import openai
import uuid
import os
import sqlite3
import sys

openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise Exception("OPENAI_API_KEY is not set.")

def chat(prompt, model="gpt-3.5-turbo"):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content

def list_topics(topics: list[str]) -> str:
    if len(topics) == 1:
        return topics[0]

    if len(topics) == 2:
        return f"{topics[0]} and {topics[1]}"

    return f"{', '.join(topics[:-1])}, and {topics[-1]}"


def generate(topics: list[str]) -> str:
    prompt = f"Come up with some topics that a user with a blog called {topics[0]} might write. Give me a blog post formatted in markdown about those topics. Do not output the topics. Just output the markdown for the blog post."
    return chat(prompt)

def get_title(body: str) -> str:
    return body.split("\n")[0].replace("# ", "")

def slugify(title: str) -> str:
    return title.lower().replace(" ", "-")

def sql_string_escape(body: str) -> str:
    return body.replace("'", "''")

if __name__ == "__main__":
    blog_id = sys.argv[1]
    topics = sys.argv[2:]

    if len(topics) == 0:
        raise Exception("Please provide at least one topic.")

    post = generate(topics)

    id = uuid.uuid4()

    print(f"""INSERT INTO "post" (id, blog_id, title, slug, body)
VALUES
    ('{id}', '{blog_id}', '{get_title(post)}', '{slugify(get_title(post))}', '{sql_string_escape(post)}');""")

