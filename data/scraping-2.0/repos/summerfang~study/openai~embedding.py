from openai import OpenAI

client = OpenAI()

embedding = client.embeddings.create(
    model = "text-embedding-ada-002",
    input = "unit test is important",
)

print(embedding.data[0].embedding)