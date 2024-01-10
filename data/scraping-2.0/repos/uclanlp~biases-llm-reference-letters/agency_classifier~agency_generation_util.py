import re
import random
import torch
import openai
from ratelimiter import RateLimiter
from retrying import retry

AGENCY_DATASET_GEN_PROMPTS = {
    'You will rephrase a biography two times to demonstrate agentic and communal language traits respectively. "agentic" is defined as more achievement-oriented, and "communal" is defined as more social or service-oriented. The paragraph is: "{}"'
}

# # Uncomment this part and fill in your OpenAI organization and API key to query ChatGPT's API
# openai.organization = $YOUR_ORGANIZATION$
# openai.api_key = $YOUR_API_KEY$

# To avoid exceeding rate limit for ChatGPT API
@retry(stop_max_attempt_number=10)
@RateLimiter(max_calls=20, period=60)
def generate_response_fn(utt):
    prompt = random.sample(AGENCY_DATASET_GEN_PROMPTS, 1)[0]  # .format(utt)
    utt = " ".join([prompt, utt])
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": utt}]
    )
    # print('ChatGPT: {}'.format(response["choices"][0]["message"]["content"].strip()))
    return response["choices"][0]["message"]["content"].strip()
