import openai
from pymongo import MongoClient
import numpy as np
import faiss 
import pickle
from pprint import pprint as pp
import re
from configsecrets import openai_apikey, mongoConnection

openai.api_key = openai_apikey
client = MongoClient(mongoConnection['host'], mongoConnection['port'],appName=mongoConnection['appName'])
client.admin.authenticate(mongoConnection['login'],mongoConnection['password'])
db = client[mongoConnection["db"]]
coll = db[mongoConnection["collection"]]

def getEmbeddingText(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embedding_vector = response["data"][0]["embedding"] #type: ignore
    return embedding_vector

def retrieveEmbeddingsMongo(search_term: dict):
    """
    Parameters:
        search_term(dict): MongoDB search term
    
    Returns:
        numpy array of embeddedings for each result of the MongoDB search term
        Id array which indexes the embeddings with the MongoDB objectID 
    """
    subset = coll.find(search_term)
    vectors = []
    id_vector = []
    for x in subset:
        embedding = x['embedding']
        obj_id = x["_id"]
        id_vector.append(obj_id)
        vectors.append(embedding)
    embeddings_array:list = np.float32(vectors) 
    return embeddings_array, id_vector

def createIndex(embeddings_array: np.float64 ,id_vector: list):
    number_embeddings, embedding_dim = embeddings_array.shape
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings_array) #type: ignore
    return index, id_vector

def saveIndex(index: faiss.IndexFlat,id_vector: list,index_name: str):
    index_package = {
        "index":index,
        "id_vector":id_vector
    }
    with open(f"indices/{index_name}","wb") as f:
         pickle.dump(index_package, f)

def retrieveIndex(index_name: str):
    with open(f"indices/{index_name}","rb") as f:
        index_package = pickle.load(f)
    return index_package["index"], index_package["id_vector"]

def queryTerm(search_text:str, k_nearest:int, index: faiss.IndexFlat, id_vector:list ):
    embedding_vector = getEmbeddingText(search_text)
    embedding_vector = np.float32(embedding_vector)
    embedding_vector.resize(1, 1536) 
    # run the search to get the nearest neighbors
    D, I = index.search(embedding_vector, k_nearest) 
    print(D[:k_nearest])
    results = getSearchResults(I, id_vector, k_nearest) #type:ignore
    return results

def remove_formatting(markdown:str) -> str:
    """
    Used to remove certain tokens from documents before sending them to gpt, in order to reduce token usage
    """
    pattern = r"[^\w\s\n*’'.,;:!?-]"
    return re.sub(pattern, "", markdown)

def getSearchResults(I, id_vector:list, k_nearest:int):    
    object_ids = []
    for i in I[0][0:k_nearest]:
        # id vector is an ordered list of the object ids, I contains the indicies in the id_vector, 
        # which lead to the desired object IDs
        object_ids.append(id_vector[i])
    search_results = coll.find({ '_id': { '$in': object_ids}},{'embedding': 0,"_id":0})
    results = []
    for s in search_results:
        clean_text = remove_formatting(s["text"])
        s["text"] = clean_text
        results.append(s)
    return results


if __name__ == "__main__":
    INDEX_NAME = "complete_index"
    embeddings_array, id_vector = retrieveEmbeddingsMongo({})
    index, id_vector = createIndex(embeddings_array,id_vector)
    saveIndex(index,id_vector,INDEX_NAME)

    index, id_vector = retrieveIndex(INDEX_NAME)
    results = queryTerm("how to create a graph which describes network topology?",5,index,id_vector)
    for r in results:
        pp(r)
    