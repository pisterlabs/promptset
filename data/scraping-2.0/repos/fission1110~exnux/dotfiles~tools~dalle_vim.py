#!/usr/bin/env python3
import os
import openai
import sys

prompt = sys.stdin.read().strip()

# read api key from file ~/.config/openai/api_key
with open(os.path.expanduser("~/.config/openai/api_key")) as f:
    openai.api_key = f.read().strip()

response = openai.Image.create(
  prompt=prompt,
  n=1,
  size="1024x1024"
)

image_url = response["data"][0]["url"]

# open url with chromium
os.system("chromium " + image_url)
print(prompt)
