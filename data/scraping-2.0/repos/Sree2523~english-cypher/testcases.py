import json
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
openai.api_key = "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"


# def get_most_similar_queries(user_input):
#     traindoc_path = r'C:\Users\srees\OneDrive - iitkgp.ac.in\Desktop\BangDB\chatbot\traindoc.json'
#     with open(traindoc_path, 'r') as file:
#         traindoc = json.load(file)

#     queries = [data['prompt'] for data in traindoc]
#     queries.extend(user_input)  # Add all user inputs to the queries list

#     vectorizer = TfidfVectorizer()
#     query_vectors = vectorizer.fit_transform(queries)

#     similarity_matrix = cosine_similarity(query_vectors[-len(user_input):], query_vectors[:-len(user_input)])
#     most_similar_indices = np.argsort(similarity_matrix, axis=1)[:, -1]

#     most_similar_queries = [traindoc[index]['completion'] for index in most_similar_indices]
#     return most_similar_queries

# with open(r'C:\Users\srees\OneDrive - iitkgp.ac.in\Desktop\BangDB\chatbot\train.json', 'r') as train_file:
#     user_input = json.load(train_file)

# most_similar_queries = get_most_similar_queries(user_input)
# i=1
# for query in most_similar_queries:
#     print("Most Similar Query",i,":")
#     print(query)
#     print()
#     i=i+1


def get_most_similar_queries(user_input):
    traindoc_path = r'C:\Users\srees\OneDrive - iitkgp.ac.in\Desktop\BangDB\chatbot\traindoc.json'
    with open(traindoc_path, 'r') as file:
        traindoc = json.load(file)

    queries = [data['prompt'] for data in traindoc]
    queries.append(user_input)

    vectorizer = TfidfVectorizer()
    query_vectors = vectorizer.fit_transform(queries)

    similarity_matrix = cosine_similarity(query_vectors[-1], query_vectors[:-1])
    most_similar_indices = np.argsort(similarity_matrix, axis=1)[:, -1]

    most_similar_queries = [traindoc[index]['completion'] for index in most_similar_indices]
    return most_similar_queries

user_input = "get hospital address"

most_similar_queries = get_most_similar_queries(user_input)
for query in most_similar_queries:
    print("Most Similar Query:", query)




# def get_most_similar_queries(user_inputs):
#     traindoc_path = r'C:\Users\srees\OneDrive - iitkgp.ac.in\Desktop\BangDB\chatbot\traindoc.json'
#     with open(traindoc_path, 'r') as file:
#         traindoc = json.load(file)

#     queries = [data['prompt'] for data in traindoc]
#     queries.extend(user_inputs)  # Add all user inputs to the queries list

#     vectorizer = TfidfVectorizer()
#     query_vectors = vectorizer.fit_transform(queries)

#     similarity_matrix = cosine_similarity(query_vectors[-len(user_inputs):], query_vectors[:-len(user_inputs)])
#     most_similar_indices = np.argsort(similarity_matrix, axis=1)[:, -1]

#     most_similar_queries = [traindoc[index]['completion'] for index in most_similar_indices]
#     return most_similar_queries

# user_inputs = ["get the first name and family name of patients", "list no of patients"]

# most_similar_queries = get_most_similar_queries(user_inputs)
# for i, query in enumerate(most_similar_queries):
#     print("Most Similar Query for Input", i + 1, ":", query)












