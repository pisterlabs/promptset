"""Get Windsor.ai data using GPT-3 to convert questions into API queries."""

from io import StringIO
import openai
import os
import sys

WINDSOR_API_KEY = os.getenv("WINDSOR_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

question = " ".join(sys.argv[1:])

prompt = f"""
Your task is to answer questions correctly. You have access to the Windsor.ai HTTP API, so if you are not able to answer a question from memory, you can write a python snippet to get data from Windsor.ai that will answer the question. Always write your answer as a valid python program, with helpful comments.

Begin.

Question: What was the Facebook total cost of the first two weeks of september
Answer:
```
import requests
# Get the Facebook costs of the first 14 days in september
res = requests.get("https://connectors.windsor.ai/facebook?fields=spend&date_from=2022-09-01&date_to=2022-09-14&api_key=THE_API_KEY")
print(res.json()["data"])
```

Question: How many impressions did each Google Ads campaign have in march?
Answer:
```
import requests
# Get the Google Ads impressions per campaign in march
res = requests.get("https://connectors.windsor.ai/google_ads?fields=campaign,impressions&date_from=2022-03-01&date_to=2022-03-31d&api_key=THE_API_KEY")
print(res.json()["data"])
```

Question: {question}
Answer:
```
"""


response = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    temperature=0,
    max_tokens=512,
    stop="```",
)

code = response.choices[0].text.strip().replace("THE_API_KEY", WINDSOR_API_KEY)
# print(code)

STDOUT = sys.stdout
sys.stdout = output = StringIO()
exec(code)
sys.stdout = STDOUT
print("Answer: " + output.getvalue())