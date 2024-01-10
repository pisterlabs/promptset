import os
import numpy as np
import openai
import pandas as pd

from typing import Iterator
from create_index import get_pinecone, load_data

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
EMBEDDING_MODEL = "text-embedding-ada-002"

def init_openai():
    openai.api_key = OPENAI_API_KEY

class BatchGenerator:
    """Models a simple batch generator that make chunks out of an input DataFrame"""
    def __init__(self, batch_size: int = 10) -> None:
        self.batch_size = batch_size

    def splits_num(self, elements: int) -> int:
        """Determines how many chunks DataFrame contains"""
        return round(elements / self.batch_size)

    def to_batches(self, df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Makes chunks out of an input DataFrame"""
        splits = self.splits_num(df.shape[0])
        if splits <= 1:
            yield df
        else:
            for chunk in np.array_split(df, splits):
                yield chunk
    
    __call__ = to_batches

def upsert_data():
    init_openai()
    
    df_batcher = BatchGenerator(300)
    index = get_pinecone()
    post_df = load_data('data/blog_posts.csv')

    for batch_df in df_batcher(post_df):
        print(batch_df.content_vector)
        index.upsert(vectors=zip(batch_df.vector_id, batch_df.content_vector), namespace='content')
    
    index.describe_index_stats()


if __name__ == "__main__":
    upsert_data()