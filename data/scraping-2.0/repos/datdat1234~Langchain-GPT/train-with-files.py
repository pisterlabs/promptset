import os
import sys

import constants
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator

os.environ["OPENAI_API_KEY"] = constants.APIKEY

loader = DirectoryLoader("./data", glob="*.txt")

print(loader)