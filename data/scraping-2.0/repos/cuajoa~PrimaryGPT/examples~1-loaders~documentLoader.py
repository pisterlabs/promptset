import openai
import pathlib
import sys

_parentdir = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
print (_parentdir)

from scripts.config import Config

cfg = Config()

openai.api_key = cfg.openai_api_key
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("ejemplos/docs/api.pdf")
pages = loader.load()

print(len(pages))

page = pages[0]
print(page.page_content[:500])
print(page.metadata)