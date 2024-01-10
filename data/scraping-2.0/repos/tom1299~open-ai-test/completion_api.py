import os
import openai
import logging

# Activate request logging support.
# Caution: Might print the openai key to the console
import http.client as http_client

if os.getenv("DEBUG"):
    http_client.HTTPConnection.debuglevel = 1
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

openai.debug = True
openai.api_key = os.getenv("OPENAI_API_KEY")

# Let chatgpt be creative by adding temperature to the completion of a very famous quote
for temperature in [1, 3, 5, 7, 9]:
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt="To be or not to be",
        temperature=temperature/10,
        max_tokens=256
    )

    for choice in response["choices"]:
        print(f"Temperature: {temperature/10}")
        print(f"Text: {choice['text']}")

