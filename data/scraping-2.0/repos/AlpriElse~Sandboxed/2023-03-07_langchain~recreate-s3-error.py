from langchain.document_loaders import S3DirectoryLoader
from dotenv import load_dotenv
load_dotenv()

import tempfile
tempfile.TemporaryDirectory()

loader = S3DirectoryLoader("langchain-debugging-test", prefix="folder")
print(loader.load())