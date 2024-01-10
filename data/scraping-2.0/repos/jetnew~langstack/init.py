import os
import openai
from dotenv import load_dotenv
from pathlib import Path
import shutil


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


def init_dir(dir, clear=True):
    if clear and os.path.exists(dir) and os.path.isdir(dir):
        shutil.rmtree(dir)
    Path(dir).mkdir(parents=True, exist_ok=True)
    return dir