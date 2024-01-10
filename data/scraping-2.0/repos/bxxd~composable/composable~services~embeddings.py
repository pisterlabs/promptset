import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
import logging

log = logging.getLogger(__name__)


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, engine="text-embedding-ada-002"):
    if not text:
        log.info(f"get_embedding: text is empty")
        return None
    log.info(f"get_embedding...")
    text = text.replace("\n", " ")
    result = openai.Embedding.create(input=[text], model=engine)
    # log.info(f"get_embedding: {result}")
    return result["data"][0]["embedding"]
