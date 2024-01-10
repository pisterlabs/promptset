

# Get all interests in Airtable


# Check embeddings DB
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, engine="text-similarity-davinci-001") -> list[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], engine=engine)["data"][0]["embedding"]


embedding = get_embedding("Sample query text goes here", engine="text-search-ada-query-001")
print(len(embedding))

# Retrieve similar interests

# Generate filter view:
# https://airtable.com/shr3xBd588LnmbEqP?filterHasAnyOf_Interests=grantmaking&filterHasAnyOf_Interests=aerospace