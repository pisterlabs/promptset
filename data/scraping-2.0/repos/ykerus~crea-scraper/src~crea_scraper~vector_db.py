import os
import time

import openai
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from crea_scraper.data import (get_course_documents_for_search,
                               load_course_data, prepare_for_search)

openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_version = os.environ["OPENAI_API_VERSION"]
openai.api_key = os.environ["OPENAI_API_KEY"]


def create_vector_db(
    data_for_seach: pd.DataFrame, save_path: str, save: bool = True
) -> FAISS:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        chunk_size=1,
    )
    courses = get_course_documents_for_search(data_for_seach)
    db = FAISS.from_documents(courses, embeddings)
    if save:
        db.save_local(save_path)
    return db


def load_vector_db(path: str) -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)
    return FAISS.load_local(path, embeddings)


if __name__ == "__main__":
    start_time = time.time()

    course_data = load_course_data("output/course_data.csv")
    data_for_search = prepare_for_search(course_data)
    vector_db = create_vector_db(data_for_search, save_path="output/vector_db")

    print("--- %s seconds ---" % (time.time() - start_time))
