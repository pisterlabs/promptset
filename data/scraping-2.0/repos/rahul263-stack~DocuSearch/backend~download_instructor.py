from os import getenv
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceInstructEmbeddings

load_dotenv() 

instructor_model_name = getenv("INSTRUCTOR_MODEL_NAME")
cache_folder = getenv("CACHE_FOLDER")

instructorEmbeddings = HuggingFaceInstructEmbeddings(model_name=instructor_model_name, model_kwargs={"device": "cuda"}, cache_folder=cache_folder)
exit()