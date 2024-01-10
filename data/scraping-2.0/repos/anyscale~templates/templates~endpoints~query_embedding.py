import openai

client = openai.OpenAI(
    base_url = "http://localhost:8000/v1",
    api_key = "NOT A REAL KEY"
)

# Note: not all arguments are currently supported and will be ignored by the backend.
embedding = client.embeddings.create(
    model="thenlper/gte-large",
    input="Your text string goes here",
)
print(embedding.model_dump())
