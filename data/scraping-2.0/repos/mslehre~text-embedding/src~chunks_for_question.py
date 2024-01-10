import os

import pandas as pd
import numpy as np
from compute_embedding import embedding_from_string
from openai.embeddings_utils import cosine_similarity
from settings import DATA_DIR

def get_k_IDs(question: str,
              embeddings_file: str,
              k: int = 5) -> list[int]:
    """Gets the IDs of the k chunks that have the highest cosine similarity
    with the embedded question. The embeddings of the chunks are given in the
    hdf5 file named after the string of 'embeddings_file' which should be
    locaded in the data directory of this repository. The function gives back a
    list that contains the IDs of the k chunks sorted decreasingly by
    similarity to the question.

    Args:
        question (str): The string of the question to compare the embeddings
            of the chunks to.
        embeddings_file (str): The relative path of the hdf5 file, that
            contains the embeddings , if one is currently in the data
            directory, since all data files should be stored there.
            Example:    For the file named "example_embeddings" in the
                        directory "data/folder" the strings should be in the
                        following format: "folder/example_embeddings"
                        (without "/" at the beginning)
            Attention: The dictionary has to have a key that contains the
                       string "embeddings" that gives the array of the
                       embeddigs and a string with "ids" that gives the ids or
                       names of the corrsponding files.
        k (int): Integer that indicates the number of chunks that are returned.

    Returns:
        list[str]: The list that contains the IDs of the k files with the
            best cosine similiarity for the question orderded from most to
            least similar.
    """
    # Check if question is given:
    if not question:
        print("No question was given. Please enter a question.")
        return [None]
    # Embed the question:
    question_embedding = embedding_from_string(string=question)

    # Get the embeddings from the hpf5 file if exists and acess is given:
    file_path = os.path.join(DATA_DIR, embeddings_file)
    if (not os.path.isfile(file_path) or not os.access(file_path, os.R_OK)):
        print("The file " + file_path + " does not exist or is not readable!")
        return [None]
    
    # Read the pd.DataFrames from the hdf5 file
    hdf = pd.HDFStore(file_path, mode='r')
    embeddings_df = pd.read_hdf(hdf, "embeddings") 
    ids_df = pd.read_hdf(hdf, "ids")
    hdf.close()

    # Check if k not bigger than the number of embeddings:
    n,_ = embeddings_df.shape
    if (k > n):
        print(f'k was given as {k} but there are {n} embeddings given. k is ' 
              + f'set to {n}.')
        k = n

    # Compute IDs of the best embeddings and return the sorted list from 
    # biggest to smallest similarity:
    inds = get_embeddings_argsort(question_embedding=question_embedding, 
                                  embeddings=embeddings_df)
    inds = [ids_df[0].iloc[i] for i in inds] 
    return inds[0:k]

def get_embeddings_argsort(question_embedding: list[float], 
                           embeddings: pd.DataFrame) -> list[int]:
    """Gets the argsort of the given embeddings from higehst to lowest cosine
    similarity with the given question.

    Args:
        question_embedding (list[float]): The embedded question to which the 
            embeddings are compared to.
        embeddings (pd.DataFrame): The pandas DataFrame containing the 
            embeddings of the chunks.   

    Returns:
        list[int]: The list that contains the indices (row numbers)of the 
            argsort of the embeddings according to the cosine similiarity for 
            the question orderd from most to least similar.
    """
    similarities = []
    n,_ = embeddings.shape
    for i in range(0,n):
        similarities.append(cosine_similarity(question_embedding, 
                                              embeddings.iloc[i].tolist()))
    
    # Return the indices of the k best embeddings, the best results have the 
    # highest cosine similarity.
    inds = np.array(similarities).argsort()[::-1]  
    return inds


def main():
    """Main to test the function that gets the k best chunks for a question.
    """
    question = "What is the publication domain of Volkmar Liebscher?"
    k = 5
    a = get_k_IDs(question, 
                embeddings_file="pub_embeddings.h5", k=k)
    print(f'Question: {question} The list with the best {k} file(s) is {a}.')

    exit(0)

if __name__ == "__main__":
    main()
