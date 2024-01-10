import os
import certifi
certifi.where()
import requests
import openai


import ssl
print(ssl.get_default_verify_paths().openssl_cafile)
"""
curl https://api.openai.com/v1/completions --ssl-no-revoke \
-H 'Content-Type:application/json' \
-H 'Authorization: Bearer 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' \
-d '{"model": "text-curie-001","prompt":"Say this is a test", "max_tokens":7,"temperature":0}'

"""

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = "I am a highly intelligent question answering bot. \
If you ask me a question that is rooted in truth, I will give you the answer. \
If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\
\n\nQ: What is human life expectancy in the United States?\
\nA: Human life expectancy in the United States is 78 years.\
\n\nQ: Who was president of the United States in 1955?\
\nA: Dwight D. Eisenhower was president of the United States in 1955.\
\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\
\n\nQ: What is the square root of banana?\
\nA: Unknown\n\nQ: How does a telescope work?\
\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\
\n\nQ: Where were the 1992 Olympics held?\
\nA: The 1992 Olympics were held in Barcelona, Spain.\
\n\nQ: How many squigs are in a bonk?\
\nA: Unknown\
\n\nQ: Where is the Valley of Kings?\nA:",

# https://platform.openai.com/docs/models/overview
# https://platform.openai.com/docs/models/gpt-3
model='gpt-3.5-turbo'
model='text-davinci-003'

response = openai.Completion.create(
    model='text-davinci-003',
    prompt=prompt,
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0.0,
    presence_penatly=0.0,
    stop=["\n"]
)
