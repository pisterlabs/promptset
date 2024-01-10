
import json
import openai
import torch
import os
import shutil
import sentence_transformers
from tqdm.notebook import tqdm
import pandas as pd
# LlamaIndex modules
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.node_parser import SentenceSplitter
from llama_index.llms import OpenAI
from llama_index.schema import MetadataMode, TextNode
from llama_index.finetuning import (
    generate_qa_embedding_pairs, EmbeddingQAFinetuneDataset, EmbeddingAdapterFinetuneEngine, SentenceTransformersFinetuneEngine)
from llama_index.embeddings import resolve_embed_model, OpenAIEmbedding, AdapterEmbeddingModel
from llama_index.embeddings.adapter_utils import TwoLayerNN
from llama_index.corpus import load_corpus

from llama_index.finetuning import (
    generate_qa_embedding_pairs,
    EmbeddingQAFinetuneDataset,
)

OPENAI_KEY=os.environ['OPENAI_KEY']

train_nodes = load_corpus([f'/kaggle/input/10k-forms/lyft_2021.pdf'], verbose=False)
val_nodes = load_corpus([f'/kaggle/input/10k-forms/uber_2021.pdf'], verbose=False)
# Creation or corpora takes about an hour without Accelerator and costs about one dollar for both sets
train_dataset = generate_qa_embedding_pairs(train_nodes, llm=OpenAI(api_key=OPENAI_KEY))
val_dataset = generate_qa_embedding_pairs(val_nodes, llm=OpenAI(api_key=OPENAI_KEY))

train_dataset = EmbeddingQAFinetuneDataset.from_json("/kaggle/input/10k-forms/train_dataset.json")
val_dataset = EmbeddingQAFinetuneDataset.from_json("/kaggle/input/10k-forms/val_dataset.json")

base_embed_model = resolve_embed_model("/mnt/ai-llm/models/UAE-Large-V1")

finetune_engine = EmbeddingAdapterFinetuneEngine(
    dataset=train_dataset,
    embed_model=base_embed_model,
    model_output_path="fine-tuned_model_03",
    model_checkpoint_path="model_ck",
    # adapter_model=adapter_model,
    epochs=25,
    verbose=False,
)
finetune_engine.finetune() # takes about an hour without Accelerator




