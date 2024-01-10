import openai
import os
from utils import io_util

def initialize_openai_api(key_file_path = 'config/openai_key.txt' ):
    openai_api_key = io_util.read_file(key_file_path)
    openai.api_key  = openai_api_key.strip()
    os.environ["OPENAI_API_KEY"] = openai.api_key
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
