import openai
import json
import os
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

load_dotenv()
openai.api_key = os.environ.get('API_KEY')


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text):
    return openai.Embedding.create(
        engine="text-embedding-ada-002",
        input=text
    )


with open('db/memory_db.json', 'r') as f:
    data = json.load(f)

with open("db/tmp_db_rows.md", "w") as f:
    json.dump('', f)

embeddings = []
entries = data
tokens = 0
for text in entries:
    print('Calculating embedding for:', text)
    response = get_embedding(text)
    tokens += response['usage']['total_tokens']
    new_embedding = {"text": text, "embedding": response["data"][0]["embedding"]}
    embeddings.append(new_embedding)
    # Write the embeddings to a file every time just in case something fails
    with open("db/tmp_db_rows.md", "a") as f:
        f.write(json.dumps(new_embedding) + '\n')

with open("db/embeddings_db.json", "w") as f:
    json.dump(embeddings, f)

print('Embeddings calculated successfully!')
print('Number of entries processed:', len(entries))
print('Total tokens used:', tokens)
