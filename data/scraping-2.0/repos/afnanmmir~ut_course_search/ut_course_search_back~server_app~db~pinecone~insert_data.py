import openai
import os
from dotenv import load_dotenv
from tqdm.auto import tqdm
import pickle
import pinecone
from time import sleep
import sys
import random

class Course():
    def __init__(self, title: str, description: str, plain_text: str):
        self.title = title
        self.description = description
        self.plain_text = plain_text

    # Create getter and setter methods for each attribute
    def get_title(self):
        return self.title
    
    def set_title(self, title):
        self.title = title
    
    def get_description(self):
        return self.description
    
    def set_description(self, description):
        self.description = description

    def get_plain_text(self):
        return self.plain_text
    
    def set_plain_text(self, plain_text):
        self.plain_text = plain_text

    # Create a method to return a dictionary of the object
    def to_dict(self):
        return {
            'title': self.title,
            'description': self.description,
            'plain_text': self.plain_text
        }
    
    def __str__(self):
        return f"Course(title={self.title}, description={self.description}, plain_text={self.plain_text})"

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENV")
EMBED_MODEL = "text-embedding-ada-002"
EMBED_DIM = 1536
batch_size = 25  # how many embeddings we create and insert at once

with open("../../data/plain_texts_courses.pkl", "rb") as f:
    courses_list = pickle.load(f)

def find_second_period(string):
    try:
        first_period = string.index(".")  # Find the index of the first period
        second_period = string.index(".", first_period + 1)  # Find the index of the second period
        return second_period
    except ValueError:
        return -1  # If no period found or only one period exists

def create_courses_list(texts):
    courses_list = []
    for text in texts:
        second_period_index = find_second_period(text)
        title = text[0:second_period_index + 1]
        description = text[second_period_index + 1:]
        courses_list.append(Course(title, description, text))
        # print(str(courses_list[-1]))
    return courses_list

# courses_list = create_courses_list(courses)

index_name = "ut-courses"

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=EMBED_DIM,
        metric="cosine",
    )

index = pinecone.Index(index_name)

delay = 1

for i in tqdm(range(0, len(courses_list), batch_size)):
    i_end = min(len(courses_list), i + batch_size) # find end of batch
    ids_batch = [str(i) for i in range(i, i_end)] # create a list of ids
    meta_batch = courses_list[i:i_end] # create a list of metadata
    texts = [course.get_plain_text() for course in meta_batch]
    try:
        res = openai.Embedding.create(input=texts, engine=EMBED_MODEL)
    except Exception as e:
        print(f"Error: {e}")
        done = False
        while not done:
            delay *= 2 * (1 + 1*random.random())
            sleep(delay)
            try:
                res = openai.Embedding.create(input=texts, engine=EMBED_MODEL)
                done = True
            except Exception as e:
                print(f"Error: {e}")
                pass
    embeds = [record["embedding"] for record in res["data"]]
    meta_batch = [{
        'title': course.get_title(),
        'description': course.get_description(),
    } for course in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    num_bytes = sys.getsizeof(to_upsert)
    # print(f"Uploading {num_bytes} bytes to Pinecone")
    index.upsert(vectors=to_upsert)
