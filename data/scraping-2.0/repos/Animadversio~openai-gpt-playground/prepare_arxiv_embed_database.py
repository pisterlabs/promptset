"""
Pipeline for preparing the arxiv embedding database
"""
import os
from os.path import join
import arxiv
import openai
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

#%
# class names of arxiv
# https://gist.github.com/jozefg/c2542f51a0b9b3f6efe528fcec90e334
CS_CLASSES = [
    'cs.' + cat for cat in [
        'AI', 'AR', 'CC', 'CE', 'CG', 'CL', 'CR', 'CV', 'CY', 'DB',
        'DC', 'DL', 'DM', 'DS', 'ET', 'FL', 'GL', 'GR', 'GT', 'HC',
        'IR', 'IT', 'LG', 'LO', 'MA', 'MM', 'MS', 'NA', 'NE', 'NI',
        'OH', 'OS', 'PF', 'PL', 'RO', 'SC', 'SD', 'SE', 'SI', 'SY',
    ]
]

MATH_CLASSES = [
    'math.' + cat for cat in [
        'AC', 'AG', 'AP', 'AT', 'CA', 'CO', 'CT', 'CV', 'DG', 'DS',
        'FA', 'GM', 'GN', 'GR', 'GT', 'HO', 'IT', 'KT', 'LO',
        'MG', 'MP', 'NA', 'NT', 'OA', 'OC', 'PR', 'QA', 'RA',
        'RT', 'SG', 'SP', 'ST', 'math-ph'
    ]
]

QBIO_CLASSES = [
    'q-bio.' + cat for cat in [
        'BM', 'CB', 'GN', 'MN', 'NC', 'OT', 'PE', 'QM', 'SC', 'TO'
    ]
]

# Which categories do we search
CLASSES = CS_CLASSES + MATH_CLASSES + QBIO_CLASSES

#%
abstr_embed_dir = "/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/openai-emb-database/Embed_arxiv_abstr"
# Name for saving the database
# database_name = "diffusion_7k"
# database_name = "LLM_5k"
# database_name = "GAN_6k"
# database_name = "VAE_2k"
# database_name = "flow_100"
# database_name = "normflow_800"
# database_name = "LLM_18k"
database_name = "diffusion_10k"
# Define the search query
# You can change this to a specific field or topic
# search_query = "cat:cs.* AND all:diffusion OR all:score-based"  # You can change this to a specific field or topic
# search_query = 'cat:cs.* AND all:"generative adversarial network" OR all:GAN'
# search_query = 'cat:cs.* AND all:"variational autoencoder" OR all:VAE'
# search_query = 'cat:cs.* AND all:"flow matching"'
# search_query = 'cat:cs.* AND all:"normalizing flow"'
# search_query = 'cat:cs.* AND all:"language model" OR all:LLM'
search_query = 'cat:cs.* AND all:diffusion OR all:score-based'
MAX_PAPER_NUM = 20000
EMBED_BATCH_SIZE = 100

# Fetch papers
search = arxiv.Search(
    query=search_query,
    max_results=MAX_PAPER_NUM,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)
#%%
# Print titles and abstracts of the latest papers
paper_collection = []
idx = 0
for paper in arxiv.Client(page_size=100, delay_seconds=5.0, num_retries=50).results(search):
    paper_collection.append(paper)
    id_pure = paper.entry_id.strip("http://arxiv.org/abs/")
    print(f"{idx} [{id_pure}] ({paper.published.date()})",
          paper.title)
    idx += 1
    # print("Abstract:", paper.summary)
    # print("Categories:", paper.categories, end=" ")
    # print("ID:", paper.entry_id, end=" ")
    # print("-" * 80)
#%%
pkl.dump(paper_collection, open(join(abstr_embed_dir, f"arxiv_collection_{database_name}.pkl"), "wb"))
df = pd.DataFrame(paper_collection)
df.to_csv(join(abstr_embed_dir, f"arxiv_collection_{database_name}.csv"))
#%%
# plot the distribution of time
time_col = [paper.published for paper in paper_collection]
plt.hist(time_col, bins=50)
plt.title("Distribution of publication time")
plt.title(f"Publication time distribution for {database_name}\n{search_query}")
plt.ylabel("count")
plt.xlabel("time")
plt.savefig(join(abstr_embed_dir, f"arxiv_time_dist_{database_name}.png"))
plt.show()

#%%
import datetime
# plot the distribution of time
time_col = [paper.published for paper in paper_collection]
plt.hist(time_col, bins=200)
plt.title("Distribution of publication time")
plt.title(f"Publication time distribution for {database_name}\n{search_query}")
plt.ylabel("count")
plt.xlabel("time")
plt.xlim([datetime.datetime(2017, 1, 1), datetime.datetime(2024, 1, 1)])
plt.savefig(join(abstr_embed_dir, f"arxiv_time_dist_{database_name}_zoom.png"))
plt.show()
#%%
plt.hist(time_col, bins=200)
plt.title("Distribution of publication time")
plt.title(f"Publication time distribution for {database_name}\n{search_query}")
plt.ylabel("count")
plt.xlabel("time")
plt.xlim([datetime.datetime(2020, 1, 1), datetime.datetime(2024, 1, 1)])
# rotate the xticks
plt.xticks(rotation=45)
plt.savefig(join(abstr_embed_dir, f"arxiv_time_dist_{database_name}_zoomplus.png"))
plt.show()
#%%
def entry2string(paper):
    id_pure = paper.entry_id.strip("http://arxiv.org/abs/")
    return f"[{id_pure}] Title: {paper.title}\nAbstract: {paper.summary}\nDate: {paper.published}"


embedstr_col = [entry2string(paper) for paper in paper_collection]
#%%
# embed all the abstracts
client = openai.OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
#%%
embedding_col = []
for i in tqdm(range(0, len(embedstr_col), EMBED_BATCH_SIZE)):
    embedstr_batch = embedstr_col[i:i + EMBED_BATCH_SIZE]
    response = client.embeddings.create(
        input=embedstr_batch,
        model="text-embedding-ada-002"
    )
    embedding_col.extend(response.data)

#%%
# save the embeddings
# pkl.dump(embedding_col, open("arxiv_embedding_7k.pkl", "wb"))
# format as array
embedding_arr = np.stack([embed.embedding for embed in embedding_col])
pkl.dump([embedding_arr, paper_collection],
         open(join(abstr_embed_dir, f"arxiv_embedding_arr_{database_name}.pkl"), "wb"))

