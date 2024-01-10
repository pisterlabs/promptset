from tqdm.auto import tqdm  # this is our progress bar
import openai
import pinecone
import os

def get_data():
    with open('./data/data.txt', 'r', encoding='utf-8') as input_file:
        content = input_file.read()
        blocks = content.split('\n\n')
        blocks = [block.strip() for block in blocks if block.strip()]
        return blocks


data = get_data() 

openai.api_key = os.getenv("OPENAI_API_KEY")

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

MODEL = "text-embedding-ada-002"

res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=MODEL
)

embeds = [record['embedding'] for record in res['data']]

if 'openai' not in pinecone.list_indexes():
    pinecone.create_index('openai', dimension=len(embeds[0]))

# connect to index
index = pinecone.Index('openai')

batch_size = 4  # process everything in batches of 32
for i in tqdm(range(0, len(data), batch_size)):
    # set end position of batch
    i_end = min(i+batch_size, len(data))
    # get batch of lines and IDs
    lines_batch = data[i: i+batch_size]
    ids_batch = [str(n) for n in range(i, i_end)]
     # create embeddings
    res = openai.Embedding.create(input=lines_batch, engine=MODEL)
    embeds = [record['embedding'] for record in res['data']]
    # prep metadata and upsert batch
    meta = [{'text': line} for line in lines_batch]
    to_upsert = zip(ids_batch, embeds, meta)
    # upsert to Pinecone
    index.upsert(vectors=list(to_upsert))