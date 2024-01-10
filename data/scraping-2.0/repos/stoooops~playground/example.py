#!/usr/bin/env python3

#
# This file is for https://coryeth.substack.com/publish/post/106850995
#

import sys

import openai

prompt = sys.argv[1].replace("\\", "")
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    temperature=0,
    max_tokens=100,
)

output = response.choices[0].text
print(f"{prompt}{output}")
