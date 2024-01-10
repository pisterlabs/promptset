import time
import random
from openai import RateLimitError
def make_request_with_retry(api_call, max_retries=5):
    for i in range(max_retries):
        try:
            return api_call()
        except RateLimitError:
            wait_time = (2 ** i) + random.random()
            time.sleep(wait_time)
    raise Exception("Still hitting rate limit after max retries")