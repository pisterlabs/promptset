import openai
import pinecone
from datasets import load_dataset


openai.api_key = "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"
# get API key from top-right dropdown on OpenAI website

print(openai.Engine.list())  # check we have authenticated

MODEL = "text-embedding-ada-002"

res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=MODEL
)
print(res)

embeds = [record['embedding'] for record in res['data']]
print(embeds)


# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key="83af22f9-ec60-484a-8eba-3519c69b251f",
    environment="eu-west4-gcp"  # find next to API key in console
)

# check if 'openai' index already exists (only create index if not)
if 'openai' not in pinecone.list_indexes():
    pinecone.create_index('openai', dimension=len(embeds[0]))
# connect to index
index = pinecone.Index('openai')

print(index)

# load the first 1K rows of the TREC dataset
trec = load_dataset('trec', split='train[:1000]')

from tqdm.auto import tqdm  # this is our progress bar

batch_size = 32  # process everything in batches of 32
for i in tqdm(range(0, len(trec['text']), batch_size)):
    # set end position of batch
    i_end = min(i+batch_size, len(trec['text']))
    # get batch of lines and IDs
    lines_batch = trec['text'][i: i+batch_size]
    ids_batch = [str(n) for n in range(i, i_end)]
    # create embeddings
    res = openai.Embedding.create(input=lines_batch, engine=MODEL)
    embeds = [record['embedding'] for record in res['data']]
    # prep metadata and upsert batch
    meta = [{'text': line} for line in lines_batch]
    to_upsert = zip(ids_batch, embeds, meta)
    # upsert to Pinecone
    index.upsert(vectors=list(to_upsert))

query = "What caused the 1929 Great Depression?"

xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']
res = index.query([xq], top_k=5, include_metadata=True)
for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")