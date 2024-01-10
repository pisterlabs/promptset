from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
from config import openai_key
import openai
import numpy as np
import pickle
import json
import os

openai.api_key = openai_key



##### Loading data into server memory ########
data_dir = "data"
with open(os.path.join(data_dir, 'embedding_matrix.pkl'), 'rb') as embedding_file:
    embedding_matrix = pickle.load(embedding_file)

# Load data dictionary from pickle file
with open(os.path.join(data_dir, 'index_to_data_dict.pkl'), 'rb') as data_dict_file:
    course_data_dict = pickle.load(data_dict_file)

with open(os.path.join(data_dir, 'data_to_index_dict.pkl'), 'rb') as data_to_index_file:
   data_to_index_dict = pickle.load(data_to_index_file)

acad_level_to_indices_map = {}

for level in ['Undergraduate', 'Graduate', 'Law', 'Graduate Business', 'Medical School', 'Non-Credit']:
   filename = os.path.join(data_dir, f"{level}_indices.pkl")
   with open(filename, 'rb') as f:
      acad_level_to_indices_map[level] = pickle.load(f)
   
with open(os.path.join(data_dir, "latest_sem_indices.pkl"), 'rb') as f:
   latest_semester_indices = pickle.load(f)
##### End of loading data into server memory ########



def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


def generate_filtered_embedding_matrix(academic_level_filter, semester_filter):
   # no filtering needed if the default filters are used
   if academic_level_filter == "all" and semester_filter == "all":
      return embedding_matrix, [i for i in range(len(embedding_matrix))]

   original_indices = set([i for i in range(len(embedding_matrix))])
   
   if academic_level_filter != "all":
      original_indices &= acad_level_to_indices_map[academic_level_filter]
   
   if semester_filter == "latest":
      original_indices &= latest_semester_indices

   original_indices = list(original_indices)
   filtered_embedding_matrix = embedding_matrix[original_indices]
   return filtered_embedding_matrix, np.array(original_indices)


def cosine_similarity_search(query_vector, embedding_matrix):
    similarities = np.dot(embedding_matrix, query_vector) / (np.linalg.norm(embedding_matrix, axis=1) * np.linalg.norm(query_vector))
    return similarities


def get_top_n_data_without_filters(query_vector, n=10):
   # if there are no filters, just use the original embedding matrix
   similarities = cosine_similarity_search(query_vector, embedding_matrix)

   top_n_indices = np.argsort(similarities)[::-1][:n]
   top_n_data = [course_data_dict[index] for index in top_n_indices]
   
   # add the similarity scores as values in the dictionaries
   for i in range(n):
      matrix_index = top_n_indices[i]
      top_n_data[i]["similarity_score"] = similarities[matrix_index]
   return top_n_data


def get_top_n_data_with_filters(query_vector, academic_level_filter="all", semester_filter="all", n=10):
   filtered_embedding_matrix, original_indices = generate_filtered_embedding_matrix(academic_level_filter, semester_filter)
   similarities = cosine_similarity_search(query_vector, filtered_embedding_matrix)

   top_n_filtered_indices = np.argsort(similarities)[::-1][:n]
   top_n_original_indices = original_indices[top_n_filtered_indices]
   top_n_data = [course_data_dict[index] for index in top_n_original_indices]

   # add the similarity scores as values in the dictionaries
   for i in range(min(n, len(top_n_data))):
      matrix_index = top_n_filtered_indices[i]
      top_n_data[i]["similarity_score"] = similarities[matrix_index]
   return top_n_data


def get_top_n_data(query_vector, academic_level_filter="all", semester_filter="all", n=10):
   if academic_level_filter == "all" and semester_filter == "all":
      return get_top_n_data_without_filters(query_vector, n=n)
   else:
      return get_top_n_data_with_filters(query_vector, academic_level_filter, semester_filter, n)


def get_top_search_results_json(query, academic_level_filter ="all", semester_filter="all",  n=10):
   query_vector = get_embedding(query, model='text-embedding-ada-002')
   top_n_data = get_top_n_data(query_vector, academic_level_filter=academic_level_filter, semester_filter=semester_filter, n=n)
   return json.dumps(top_n_data)


def get_similar_courses(mnemonic, catalog_number, academic_level_filter="all", semester_filter="all", n=10):
   id_tuple = (mnemonic, str(catalog_number))
   print(id_tuple)
   if not id_tuple in data_to_index_dict.keys():
      print("no matching")
      return json.dumps([])   # no matching courses
   index = data_to_index_dict[id_tuple]
   query_vector = embedding_matrix[index]
   top_n_data = get_top_n_data(query_vector, academic_level_filter=academic_level_filter, semester_filter=semester_filter, n=n+1)
   return json.dumps(top_n_data[1:])
