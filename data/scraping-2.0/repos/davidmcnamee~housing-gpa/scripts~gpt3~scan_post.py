import asyncio
import os
from typing import List
from scripts.common.types import FacebookPost

import openai
from dotenv import load_dotenv
import chevron

load_dotenv()

async def scan(posts: List[FacebookPost]):
  openai.api_key = os.getenv("OPENAI_API_KEY")
  data = "\n\n".join(post.main_text + "\n" + post.bottom_text for post in posts)
  mustache_file = os.path.join(os.path.dirname(__file__) + "/gpt3-template.mustache")
  with open(mustache_file, 'r') as f:
    prompt = chevron.render(f, {"data": data})
    print(prompt)
  # response = openai.Completion.create(engine="babbage", prompt=prompt, max_tokens=5)
  # print(response)
