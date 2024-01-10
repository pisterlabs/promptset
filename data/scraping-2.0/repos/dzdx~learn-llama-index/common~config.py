import os

# import openai
# HTTP_PROXY = "http://127.0.0.1:7890"
# openai.proxy = {"http": HTTP_PROXY, "https": HTTP_PROXY}

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
DEBUG = True
LLM_CACHE_ENABLED = True

OPENAI_API_KEY = ''
if OPENAI_API_KEY:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

data_dir = os.path.join(ROOT_PATH, 'data')
index_dir = os.path.join(ROOT_PATH, 'index')

ROUTE_TODO = True
