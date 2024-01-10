
from typing import List, Dict
import pandas as pd
import os
import numpy as np
import openai
from publang.search.embed import embed_pmc_articles
from publang.extract.extract import search_extract
from publang.extract.templates import ZERO_SHOT_MULTI_GROUP


def clean_gpt_demo_predictions(predictions):
    # Clean known issues with GPT demographics predictions

    predictions = predictions.copy()
    
    predictions = predictions.fillna(value=np.nan)
    predictions['group_name'] = predictions['group_name'].fillna('healthy')

    # If group name is healthy, blank out diagnosis
    predictions.loc[predictions.group_name == 'healthy', 'diagnosis'] = np.nan
    predictions = predictions.replace(0.0, np.nan)

    # Drop rows where count is NA
    predictions = predictions[~pd.isna(predictions['count'])]

    # Set group_name to healthy if no diagnosis
    predictions.loc[(predictions['group_name'] != 'healthy') & (pd.isna(predictions['diagnosis'])), 'group_name'] = 'healthy'

    # If no male count, substract count from female count columns
    ix_male_miss = (pd.isna(predictions['male count'])) & ~(pd.isna(predictions['female count']))
    predictions.loc[ix_male_miss, 'male count'] = predictions.loc[ix_male_miss, 'count'] - predictions.loc[ix_male_miss, 'female count']

    # Same for female count
    ix_female_miss = (pd.isna(predictions['female count'])) & ~(pd.isna(predictions['male count']))
    predictions.loc[ix_female_miss, 'female count'] = predictions.loc[ix_female_miss, 'count'] - predictions.loc[ix_female_miss, 'male count']

    return predictions

def extract_gpt_demographics(
        articles: List[Dict] = None,
        output_path: str = None,
        embeddings: pd.DataFrame = None,
        api_key: str = None,
        embedding_model_name: str = 'text-embedding-ada-002', 
        min_tokens: int = 30, 
        max_tokens: int = 4000,
        search_query: str = None,
        template: dict = None,
        extraction_model_name: str = 'gpt-3.5-turbo',
        num_workers: int = 1,
        embeddings_path: str = None,
        ) -> (pd.DataFrame, pd.DataFrame):
    """Extract participant demographics from a list of articles using OpenAI's API.

    Args:
        articles (list): List of articles. Each article is a dictionary with keys 'pmcid' and 'text'.
        output_path (str): Path to csv file to save the predictions to. If file exists, the predictions will
            be loaded from the file, and extraction will start from the last article in the file.
        embeddings_path (str): Path to parquet file to save the embeddings to. If None, the embeddings will 
            not be saved
        api_key (str): OpenAI API key. If None, the key will be read from the OPENAI_API_KEY environment variable.
        embedding_model_name (str): Name of the OpenAI embedding model to use.
        min_tokens (int): Minimum number of tokens per chunk.
        max_tokens (int): Maximum number of tokens per chunk.
        search_query (str): Query to use for semantic search.
        heuristic_strategy (str): Heuristic strategy to use for the extraction.
        template (dict): Template to use for the extraction (see templates.py).
        extraction_model_name (str): Name of the OpenAI model to use for the extraction.
        num_workers (int): Number of workers to use for parallel processing.
    Returns:
        pd.DataFrame: Dataframe containing the extracted values.
        pd.DataFrame: Dataframe containing the chunked embeddings for each article.
    """

    if api_key is not None:
        openai.api_key = api_key
    else:
        openai.api_key = os.environ.get('OPENAI_API_KEY', None)

    unique_pmcids = set([a['pmcid'] for a in articles])

    embeddings = None
    if embeddings_path is not None and os.path.exists(embeddings_path):
        # Load embeddings from file, but only for articles in input
        embeddings = pd.read_parquet(embeddings_path, filters=[('pmcid', 'in', unique_pmcids)])

    if embeddings is None:
        if articles is None:
            raise ValueError('Either articles or embeddings must be provided.')
        print('Embedding articles...')
        embeddings = embed_pmc_articles(
            articles, embedding_model_name, min_tokens, max_tokens, num_workers=num_workers)
        embeddings = pd.DataFrame(embeddings)

        if embeddings_path is not None:
            embeddings.to_parquet(embeddings_path, index=False)

    if search_query is None:
        search_query = 'How many participants or subjects were recruited for this study?' 

    if template is None:
        template = ZERO_SHOT_MULTI_GROUP

    embeddings = embeddings[embeddings.section_0 == 'Body']

    print('Extracting demographics...')
    predictions = search_extract(
        embeddings, search_query, **template, 
        model_name=extraction_model_name, 
        num_workers=num_workers,
        output_path=output_path
        )

    return predictions