import os

from dotenv import load_dotenv
from classes.NuvemIndex import NuvemIndex
import openai

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

NuvemIndex().indexDocs()