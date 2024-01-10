import os
import pathlib
from dotenv import load_dotenv, find_dotenv

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

# load environment variables
load_dotenv(find_dotenv())

# root directory
ROOT = pathlib.Path(__file__).parent.absolute()

# source directory
SOURCE_PATH = os.path.join(ROOT, "source_documents")

# database directory
DATABASE_PATH = os.path.join(ROOT, "cache_data")

# embeddings
embeddings = OpenAIEmbeddings()
# embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

# data preprocessing
chunk_size = 500
chunk_overlap = 100

# data version
data_save_time = "2023-07-23_12-05-35"
