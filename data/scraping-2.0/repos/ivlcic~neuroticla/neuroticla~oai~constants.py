import os
import openai

EMBEDDING_ENCODING = 'cl100k_base'
EMBEDDING_CTX_LENGTH = 8191

MODEL_TOKENS = {
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'gpt-3.5-turbo': 4096,
    'text-davinci-003': 4097,
    'davinci': 2049,
}

MODEL_ABBREV = {
    'gpt-4': 'g4',
    'gpt-4-32k': 'g4l',
    'gpt-3.5-turbo': 'g35',
    'text-davinci-003': 'd3',
    'davinci': 'd',
}

DEFAULT_ENCODING = 'cl100k_base'
DEFAULT_LENGTH = 2048
DEFAULT_MODEL = 'text-davinci-003'
DEFAULT_S_REPLACE = '<document>'

openai.organization = os.environ.get("OPENAI_ORG")
openai.api_key = os.environ.get("OPENAI_API_KEY")
