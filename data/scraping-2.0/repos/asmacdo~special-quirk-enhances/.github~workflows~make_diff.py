#!/usr/bin/python3
import os
# import openai
import json
import sys

# openai.api_key = os.getenv('OPENAI_API_KEY')


# TODO get issue from command
if len(sys.argv) != 3:
    raise Exception("Invocation: ./make_diff.py issue_title issue_body
title = sys.argv[1]
body = sys.argv[2]
# issue = json.loads(os.getenv("ISSUE"))


# TODO get file from command
# file = json.loads(os.getenv("FILE"))

# base_prompt = "Generate a diff of {file_to_change} that incorporates the feedback of the github issue below.

# prompt = "{base_prompt}:\n{issue['body']}"

# response = openai.Completion.create(
#     engine="text-davinci-002",
#     prompt=prompt,
#     max_tokens=500
# )
#
# print(response.choices[0].text.strip())


