from datasets import load_dataset
import pinecone
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pprint import pprint
from tqdm.auto import tqdm

EMB_DIM = 1536
METRIC = "cosine"

class KG_Builder():
    def __init__(self, PINE_API_KEY, EMBEDDING_MODEL, INDEX_NAME):
        self.embedding_model = EMBEDDING_MODEL
        pinecone.init(
            api_key=PINE_API_KEY
        )
        self.index = pinecone.Index(INDEX_NAME)


    def pc_index_upsert(self, index, ids, embeds, metadata):
        index.upsert(vectors= zip(ids, embeds, metadata))

    def pinecone_init_index(self, index_name):
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                index_name,
                dimension= EMB_DIM,
                metric=METRIC
            )
            while not pinecone.describe_index(index_name).status["ready"]:
                time.sleep(1)

    
    def process_data(self, dataset, bsz):
        dataset = dataset.to_pandas()

        for i in tqdm(range(0, len(dataset), bsz)):
            i_end = min(len(dataset), i+bsz)
            # get batch of data
            batch = dataset.iloc[i:i_end]
            # generate unique ids for each chunk
            ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
            # get text to embed
            texts = [x['chunk'] for _, x in batch.iterrows()]
            # embed text
            embeds = self.embedding_model.embed_documents(texts)
            # get metadata to store in Pinecone
            metadata = [
                {'text': x['chunk'],
                'source': x['source'],
                'title': x['title']} for i, x in batch.iterrows()
            ]
            self.pc_index_upsert(self.index, ids, embeds, metadata)



# dataset = load_dataset(
#     "jamescalam/llama-2-arxiv-papers-chunked",
#     split="train"
# )

# pinecone.init(
#     api_key=PINE_API_KEY
# )

# """Setting up index to be written to vector db"""

# index_name = 'llama-2-rag'

# if index_name not in pinecone.list_indexes():
#     pinecone.create_index(
#         index_name,
#         dimension= EMB_DIM,
#         metric=METRIC
#     )
#     while not pinecone.describe_index(index_name).status["ready"]:
#         time.sleep(1)

# index = pinecone.Index(index_name)

# # print(index.describe_index_stats())

# '''Tokenization and Text Embedding'''
# embed_model = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=API_KEY)

# # process_data(dataset, 100)
