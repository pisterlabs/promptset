from sentence_transformers import SentenceTransformer, util
import torch
import openai

# Add your own OpenAI API key
openai.api_key = ""

model = SentenceTransformer("bert-base-nli-mean-tokens")

lines = []

with open("./output_og_3.txt", "r") as f:
    for line in f:
        if line.strip() != "":
            lines.append(line.strip())

embeddings = model.encode(lines)

query = "What does Rich Harris think about type safety?"
query_embedding = model.encode(query)

hits = util.semantic_search(query_embedding, embeddings, top_k=3)
hits = hits[0]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You will be given a query sent by a user and the relevant content for the query retrieved by semantic serach. Generate a context-aware and brief response to the query that is conversationally accurate. Do not be wordy; be concise!",
        },
        {
            "role": "user",
            "content": f"QUERY: {query}, CONTENT: {lines[hits[0]['corpus_id'] - 1] + ' ' + lines[hits[0]['corpus_id']] + ' ' + lines[hits[0]['corpus_id'] + 1]}",
        },
    ],
    max_tokens=1000,
    n=1,
    stop=None,
    temperature=0.5,
)

print(response.choices[0]["message"]["content"].strip())
