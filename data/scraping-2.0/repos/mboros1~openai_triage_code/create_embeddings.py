
import openai
import chunking
import pandas as pd

# calculate embeddings
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

chunks = chunking.chunk_esi_handbook()

embeddings = []
for batch_start in range(0, len(chunks), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = chunks[batch_start:batch_end]
    print(f"Batch {batch_start} to {batch_end-1}")
    response = openai.embeddings.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response.data):
        assert i == be.index  # double check embeddings are in same order as input
    batch_embeddings = [e.embedding for e in response.data]
    embeddings.extend(batch_embeddings)

df = pd.DataFrame({"text": chunks, "embedding": embeddings})

# save document chunks and embeddings

SAVE_PATH = "chunked_esi_handbook.csv"

df.to_csv(SAVE_PATH, index=False)
