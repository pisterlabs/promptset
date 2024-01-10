from openai import OpenAI
from decouple import config


client = OpenAI(api_key=config("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-ada-002"


def get_quote_embedding(quote):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=quote,
    )
    return response


print(get_quote_embedding("Please embed this sentence for me!"))
