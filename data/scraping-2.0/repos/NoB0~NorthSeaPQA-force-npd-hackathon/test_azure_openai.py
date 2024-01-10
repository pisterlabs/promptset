from openai import AzureOpenAI
from openai.types import Embedding
from dotenv import load_dotenv
import pytest
import os
import logging
import tiktoken 
from json import JSONDecodeError
from sklearn.metrics.pairwise import cosine_similarity
from cluestar import plot_text

load_dotenv('.env.shared')
load_dotenv('/.env.secret')

test = os.getenv('OPENAI_API_KEY')
print(test)