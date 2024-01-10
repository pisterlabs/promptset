#!/usr/bin/env python3.10
import os
import sys
import openai

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG = os.getenv("OPENAI_ORG")

if not OPENAI_API_KEY or not OPENAI_ORG:
	print("You need to set values for OPENAI_API_KEY & OPENAI_ORG in .env file")
	sys.exit(1) 
else:
	openai.organization = OPENAI_ORG
	openai.api_key = OPENAI_API_KEY

def colorize(string, color):
    colors = {
        'black':'\033[30m',
        'error':'\033[31m',
        'red':'\033[31m',
        'green':'\033[32m',
        'orange':'\033[33m',
        'blue':'\033[34m',
        'purple':'\033[35m',
        'cyan':'\033[36m',
        'lightgrey':'\033[37m',
        'table':'\033[37m',
        'darkgrey':'\033[90m',
        'lightred':'\033[91m',
        'lightgreen':'\033[92m',
        'yellow':'\033[93m',
        'lightblue':'\033[94m',
        'status':'\033[94m',
        'pink':'\033[95m',
        'lightcyan':'\033[96m',
    }
    if color not in colors:
        return f"{string}"
    else:
        return f"{colors[color]} {string}\033[0m"

def ask_davinci(query):
	resp = openai.Completion.create(
	  model="text-davinci-003",
	  prompt=query,
	  max_tokens=250,
	  temperature=0.762
	)
	return resp

if __name__ == '__main__':
	while True:
		query = input(colorize("\n\n> What do you want from me?\n\n", "cyan"))
		if query.lower() in ["quit", "exit"]:
			sys.exit(0)

		resp = ask_davinci(query)
		if "choices" in resp:
			response = resp["choices"][0]
		else:
			response = colorize("> " + resp, "red")

		if "finish_reason" in response:
			finish_reason = response["finish_reason"]
			if finish_reason == "stop":
				response = colorize("\n\n> " + response["text"].replace("\n", "").replace(".", ".\n"), "lightgreen")
			else:
				print("\n\n")
				response = colorize(resp, "red")
		print(response)
