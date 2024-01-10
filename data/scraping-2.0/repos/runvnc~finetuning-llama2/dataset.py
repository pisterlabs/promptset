import sagemaker
import boto3
import time
import json

from datasets import Dataset
from langchain.document_loaders import WebBaseLoader, DirectoryLoader
from random import randint
from itertools import chain
from functools import partial
from transformers import AutoTokenizer
from sagemaker.huggingface import HuggingFace, HuggingFaceModel
from huggingface_hub import HfFolder

import init_sagemaker

def strip_spaces(doc):
    return {"text": doc.page_content.replace("  ", "")}

def chunk(sample, chunk_length=512):
    # define global remainder variable to save remainder from batches to use in next batch
    global remainder

    concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
    concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
   
    batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

    # get max number of chunks for batch
    if batch_total_length >= chunk_length:
        batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
        for k, t in concatenated_examples.items()
    }

    remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}

    result["labels"] = result["input_ids"].copy()
    return result

def sum_dataset_arrays(lm_dataset):
    total = 0
    for i in range(0,8):
        total = total + len(lm_dataset[i]['input_ids'])
    return total


def load_from_dir(path, glob="**/*"):
    loader = DirectoryLoader(path, glob)
    return process(loader)

def load_from_web(urls, model_id = "meta-llama/Llama-2-7b-hf"):
    loader = WebBaseLoader(urls)
    return process(loader)

def process(loader):
    data = loader.load()

    print("Dataset from the following URLs:",urls)
    print()
    print(data)
    print()
    print()

    stripped_data = list(map(strip_spaces, data))

    dataset = Dataset.from_list(stripped_data)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    lm_dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(dataset.features)
    ).map(
        partial(chunk, chunk_length=4096),
        batched=True,
    )

    print(f"Total number of training samples: {len(lm_dataset)}")
    print(f"Total number of training tokens: {sum_dataset_arrays(lm_dataset)}")
    return lm_dataset

#f's3://{sess.default_bucket()}/processed/llama/genai-nyc-summit/train'
def store_dataset(sess, lm_dataset, s3_path):
    training_input_path = 's3://' + sess.default_bucket() + '/' + s3_path
    lm_dataset.save_to_disk(training_input_path)

    print(f"Uploaded training dataset to: {training_input_path}")


def store_url_dataset(pretrained_model_id, s3_path, urls):
    global remainder
    # empty list to save remainder from batches to use in next batch
    remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

    dataset = load_from_web(urls, pretrained_model_id)
    (sess, llm_image, role)  = init_sagemaker.init_session()
    store_dataset(sess, dataset, s3_path)

def store_docs_dataset(pretrained_model_id, s3_path, docs_path):
    global remainder
    remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

    dataset = load_from_dir(urls, pretrained_model_id)
    (sess, llm_image, role)  = init_sagemaker.init_session()
    store_dataset(sess, dataset, s3_path)


