import json

import numpy as np
import pandas as pd
from openai.embeddings_utils import cosine_similarity

def _load_most_recent_json_file(directory):
    """Loads the most recent file in a directory as json"""
    import os
    import json
    import datetime

    # get all files in directory
    files = [f'{directory}/' + f for f in os.listdir(directory)]
    # get the most recent file
    most_recent_file = max(files, key=os.path.getctime)
    # print the filename of the most recent file
    print(f"Using resume: {most_recent_file}")
    # load the most recent file
    with open(f"{most_recent_file}", 'r') as f:
        most_recent_json = json.load(f)

    return most_recent_json

def score_job_from_dict(job_dict, resume_embedding):
    """Scores a job from a dictionary representation of a job"""
    job_embedding = np.array(json.loads(job_dict['embedding_json']))
    return int(cosine_similarity(job_embedding, resume_embedding) * 100)

def score_job_from_embedding(job_embedding, resume_embedding):
    """Scores a job from an embedding"""
    return int(cosine_similarity(job_embedding, resume_embedding) * 100)