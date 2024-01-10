import json

import numpy as np
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity

def rank_postings_from_paths(resume_path, jobs_path, n=10, pprint=True):
    df = pd.read_csv(jobs_path, index_col=0)
    # cast embedding to np array 
    df.embedding = df.embedding.apply(lambda x: np.array(json.loads(x)))

    # load resume embedding
    with open(resume_path, "r") as f:
        resume_embedding = json.load(f)
    # cast resume embedding to np array
    resume_embedding = np.array(resume_embedding)

    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, resume_embedding))

    results = df.sort_values("similarity", ascending=False).head(n)

    if pprint:
        for i, r in results.iterrows():
            out_string = r["title"] + " at " + r["company"] + " in " + r["location"] + " -- Relevance Score: " + str(int(r["similarity"] * 100))
            print(out_string)

    return results

def rank_postings_from_sources(resume_embedding, jobs_df, n=10, pprint=False):
    # cast embedding to np array 
    jobs_df.embedding = jobs_df.embedding.apply(lambda x: np.asarray(x))

    # cast resume embedding to np array
    resume_embedding = np.array(resume_embedding)

    jobs_df["similarity"] = jobs_df.embedding.apply(lambda x: cosine_similarity(x, resume_embedding))
    # apply int(x["similarity"] * 100) to similarity column
    jobs_df["similarity"] = jobs_df.similarity.apply(lambda x: int(x * 100))

    results = jobs_df.sort_values("similarity", ascending=False).head(n)

    if pprint:
        for i, r in results.iterrows():
            out_string = r["title"] + " at " + r["company"] + " in " + r["location"] + " -- Relevance Score: " + str(int(r["similarity"] * 100))
            print(out_string)

    return results

rpath = "app/jobapp/data/real_resume_embedding.json"
jpath = "app/jobapp/data/example_jobs_with_embeddings.csv"

rsrc = json.loads(open(rpath, "r").read())
jsrc = pd.read_csv(jpath, index_col=0)