import os
import sys

import constants
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import GPT4All, OpenAI

os.environ["OPENAI_API_KEY"] = constants.APIKEY

WEIGHTS_PATH = '/home/przemek/deep-learning/road-to-deep-learning/road-to-deep-learning/llm/gpt/models/' \
               'ggml-gpt4all-j-v1.3-groovy.bin'
llm = OpenAI()
#llm = GPT4All(model=WEIGHTS_PATH, verbose=True)

query = sys.argv[1]

loader = DirectoryLoader("./data", glob="*txt")
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query, llm=llm))
