# initialize embedding using promptsafe

import sys
import os
import shutil
sys.path.append("./PromptSafe")

from embedding_save import save_embedding
from prompts import prompt_dict
import openai
from scripts.key import openai_key

path = './embeddings'
if not os.path.exists(path):
    os.makedirs(path)
elif os.path.exists(path):
    shutil.rmtree(path)
    os.makedirs(path)

openai.api_key = openai_key

for key in prompt_dict:
    save_embedding(prompt_dict[key], path, key)