import openai
import tenacity
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

retry_decorator = tenacity.retry(
    stop=tenacity.stop_after_attempt(3),  # Number of retries before giving up
    wait=tenacity.wait_fixed(2),  # Time delay (in seconds) between retries
)


@retry_decorator
def get_gpt_3_5_turbo_response(**kwargs):
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", **kwargs)
    return response.choices[0].message.content
