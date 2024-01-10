import os

from dotenv import load_dotenv
from openai import OpenAI

# load .env
load_dotenv(".env")

# set globals
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"


def article_summary(article):
    client = OpenAI(
        api_key=OPEN_API_KEY,
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
              "role": "system",
              "content": "You are a news reporter."
            },
            {
                "role": "user",
                "content": f"""Please summarize this article in 4 sentences:

```text
{article}
```""",
            }
        ],
        model=OPENAI_MODEL,
    )

    return chat_completion.choices[0].message.content
