# %%
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import torch
from torch import Tensor
from rich import print
import torch
from typing import List, Tuple, Union, Callable
import numpy as np
import pandas as pd


from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import openai
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import functools
import time
import math

from dotenv import load_dotenv
load_dotenv()

from tenacity import retry, stop_after_attempt, wait_random_exponential


import google.generativeai as genai
import google.ai.generativelanguage as glm
from google.api_core import retry as gretry
from mistralai.client import MistralClient



# %%
# openai.api_key = os.getenv('OPENAI_API_KEY')
# API_KEY = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=API_KEY)
# %%
# Sentence embedding with bert
# define model
model_name = "bert-base-uncased"

# define the tokenizer and the model
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
def bert_sentence_embed(input_sentence: str, model: BertModel = bert_model, word_ave: bool = True) -> Tensor:
    """A function to generate sentence embedding using Bert

    Args:
        input_sentence (str): The sentence to embed

    Returns:
        Tensor: Tensor output for the sentence embedding
    """

    temp_list = []
    for text in input_sentence:
        input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)

            if word_ave:
                sentence_embedding = outputs.last_hidden_state.mean(dim=1)
            else:
                sentence_embedding = outputs.last_hidden_state
                sentence_embedding = sentence_embedding[0, 0, :].reshape(1, -1)
            temp_list.append(sentence_embedding)
    concatenated_sent_tensor = torch.cat(temp_list)
    # # Convert the concatenated tensor to a numpy array
    sent_numpy_array = concatenated_sent_tensor.numpy()

    # # Reshape the numpy array into a 1 by 768 array
    sent_numpy_array = sent_numpy_array.reshape(len(temp_list), temp_list[0].shape[1])
    return sent_numpy_array

# %%
# A function to create bert and other embeddings
def create_sentence_embedding(input_text: List[str],model:Union[Callable, object, str],
bert: Union[bool, None] = None, word_ave: Union[bool, None] = None, mistral: Union[bool, None] = None) -> Tuple[np.array]:
    """A function to create sentence embeddings from a list of text using
    Bert or other open source model.

    Args:
        input_text (List): List of sentences to create embeddings for.
        model (Union[Callable, object, str]): The model to be used for creating embeddings.
        bert (Union[bool, None], optional): If True, uses Bert for embedding. Defaults to None.
        word_ave (Union[bool, None], optional): If True, uses word averaging. Defaults to None.
        mistral (Union[bool, None], optional): If True, uses Mistral for embedding. Defaults to None.

    Returns:
        Tuple[np.array, np.array]: High dimensional and low dimensional vectors for all the sentence embeddings.
    """
    embed_list = []
    # model_name = 'bert-base-uncased'
    if bert:
        sent_numpy_array = bert_sentence_embed(input_text, model, word_ave)
    elif mistral:
        mistral_api_key = os.environ['MISTRAL_API_KEY']
        client = MistralClient(api_key=mistral_api_key)
        mistral_embeddings_response = client.embeddings(
        model="mistral-embed",
        input=input_text,)
        for i in range(len(mistral_embeddings_response.data)):
            embed_list.append(mistral_embeddings_response.data[i].embedding)
        sent_numpy_array = np.concatenate(embed_list).reshape(len(input_text), -1)

    else:
        for text in input_text:
            sen_emb = model.encode(text)
            embed_list.append(sen_emb)
        concatenated_sent_tensor = np.concatenate(embed_list)

        # Reshape the numpy array into a 1 by 768 array
        sent_numpy_array = concatenated_sent_tensor.reshape(len(embed_list), len(embed_list[0]))

    if len(sent_numpy_array) == 1:
        print("PCA of an array of one sample does not make sense.\nSo returning the full array.")
        return sent_numpy_array, None
    else:
        # Perform PCA for 2D visualization
        # convert the 768-dimensional array to 2-dimentional array for plotting purpose
        PCA_model = PCA(n_components=2)
        PCA_model.fit(sent_numpy_array)
        sent_low_dim_array = PCA_model.transform(sent_numpy_array)

        return sent_numpy_array, sent_low_dim_array

# %%
# A fucntion to compute cosine similarity
def compare(embeddings: np.array, idx1: int, idx2: int) -> float:
    """A function to compute cosine similarity between two embedding vectors

    Args:
        embeddings (np.array): An array of embeddings
        idx1 (int): Index of the first embedding
        idx2 (int): index of the second embedding

    Returns:
        int: The distance between the two embeddings
    """
    item1 = embeddings[idx1, :].reshape(1, -1)
    item2 = embeddings[idx2, :].reshape(1, -1)
    distance = cosine_similarity(item1, item2)
    return distance.item()

# %%

#-----------------------------------------------
# A fucntion to compute sentence embedding using OpenAI
#-----------------------------------------------

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-similarity-davinci-001", **kwargs) -> List[float]:

    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    response = openai.embeddings.create(input=[text], model=model, **kwargs)

    return response.data[0].embedding


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embeddings(
    list_of_text: List[str], model="text-similarity-babbage-001", **kwargs
) -> List[List[float]]:
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    # replace newlines, which can negatively affect performance.
    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = openai.embeddings.create(input=list_of_text, model=model, **kwargs).data
    embed_list = [d.embedding for d in data]
    embed_array = np.array(embed_list)

    # Convert to 2-dimensional vector to be able to visualize the embeddings
    PCA_model = PCA(n_components=2)
    PCA_model.fit(embed_array)
    sent_low_dim_array = PCA_model.transform(embed_array)
    return embed_array, sent_low_dim_array

    # embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# %%

#-----------------------------------------------
# A fucntion to compute sentence embedding using Google's Gemini
#-----------------------------------------------

def gembed_fn(model, input_text: List[str]) -> np.array:
    embed_list = []
    for text in input_text:
        # set the task type to semantic_similarity
        embedding = genai.embed_content(model = model, content = text,
        task_type = "semantic_similarity")["embedding"]
        embed_list.append(embedding)
    embed_array = np.array(embed_list)
     # Convert to 2-dimensional vector to be able to visualize the embeddings
    PCA_model = PCA(n_components=2)
    PCA_model.fit(embed_array)
    sent_low_dim_array = PCA_model.transform(embed_array)
    return embed_array, sent_low_dim_array

# %%
#------------------------------------------------
# get embedding in batches
#------------------------------------------------

# create a function to generate batches of data
def generate_batches(sentences: pd.Series, batch_size: int = 5) -> pd.Series:
    for i in range(0, len(sentences), batch_size):
        yield sentences[i: i + batch_size]


def encode_text_to_embedding_batched(embd_func: Callable,
        sentences: List[str],
        model: object,
        bert: bool = False,
        api_calls_per_second: float = 0.33,
        batch_size: int = 5) -> np.array:
    """A function to generate embedding in batches and respect rate limiting

    Args:
        embd_func (Callable): The function to make the APi call to do the embedding
        sentences (List[str]): List of sentences of interest
        model (object): The model that will be used to get the embedding
        bert (bool, optional): Determines if we use Bert model or not. Defaults to False.
        api_calls_per_second (float, optional): Number to use to calculate time to wait between API calls. Defaults to 0.33.
        batch_size (int, optional): Batch size. Defaults to 5.

    Returns:
        np.array: Numpy array of embeddings
    """
    # Generates batches and calls embedding API

    embeddings_list = []

    # Prepare the batches using a generator
    batches = generate_batches(sentences, batch_size)

    seconds_per_job = 1 / api_calls_per_second

    with ThreadPoolExecutor() as executor:
        futures = []
        for batch in tqdm(
            batches, total = math.ceil(len(sentences) / batch_size), position=0
        ):
            # Create a partial function with the additional arguments
            func = functools.partial(embd_func, batch, model, bert)
            futures.append(executor.submit(func))
            time.sleep(seconds_per_job)

        for future in futures:
            embeddings_list.extend(future.result()[0])

    is_successful = [
        embedding is not None for sentence, embedding in zip(sentences, embeddings_list)
    ]
    # embeddings_list_successful = np.squeeze(
    #     np.stack([embedding for embedding in embeddings_list if embedding is not None])
    # )
    embeddings_array_successful = np.vstack(embeddings_list)

    return embeddings_array_successful
