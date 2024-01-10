# import code_embedding
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# import json

# def similarity_search(query, directories=None):
#     # Default directories
#     if directories is None:
#         directories = ['db_ros2_control', 'db_ros2', 'db_webots_ros2', 'db_webots']

#     # Embeddings
#     embeddings = OpenAIEmbeddings()

#     # Load vector database for each directory
#     vectordbs = []
#     for directory in directories:
#         vectordb = Chroma(persist_directory=directory, embedding_function=embeddings)
#         vectordbs.append(vectordb)

#     # Query each vector database and concatenate results
#     results_list = []
#     for directory, vectordb in zip(directories, vectordbs):
#         results = vectordb.similarity_search(query, k=2)

#         # Convert results to JSON string
#         results_dict_list = [vars(result) for result in results]
#         results_string = json.dumps(results_dict_list)
#         results_list.append(f"Directory {directory}: {results_string}")

#     # Concatenate results into a single string
#     results_string = "\n".join(results_list)
#     return results_string


import code_embedding
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import json

def similarity_search(query, directories=None):
    # Default directories
    if directories is None:
        directories = ['db_ros2_control', 'db_ros2', 'db_webots_ros2', 'db_webots']

    # Embeddings
    embeddings = OpenAIEmbeddings()

    # Load vector database for each directory
    vectordbs = []
    for directory in directories:
        vectordb = Chroma(persist_directory=directory, embedding_function=embeddings)
        vectordbs.append(vectordb)

    # Query each vector database and concatenate results
    results_list = []
    for directory, vectordb in zip(directories, vectordbs):
        results = vectordb.similarity_search(query, k=2)

        # Only add the result to the list if it's not empty
        if results:
            # Convert results to JSON string
            results_dict_list = [vars(result) for result in results]
            results_string = json.dumps(results_dict_list)
            results_list.append(f"Directory {directory}: {results_string}")

    # Concatenate results into a single string
    results_string = "\n".join(results_list)
    return results_string

print(similarity_search("variabl;e impedance control", directories= ['db_ros2']))


