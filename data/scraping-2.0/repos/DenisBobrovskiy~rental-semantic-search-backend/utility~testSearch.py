import openai
import pymongo
import json

# this function calls OpenAI Embeddings API and returns the embedding generated for the input text
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

#Find documents in the database's collection most similar to the inputEmbedding
def findSimilarDocuments(collection, input_embedding):
    #Generate a query that uses the Atlas search index to find documents with their "openai_embedding" field
    similar_docs_cusor = collection.aggregate([
        {
            "$search": {
                "index": "vector_search_openai", #name of the search index
                "knnBeta": {
                    "vector": input_embedding, #input embedding used for the search
                    "path": "openai_embedding", #the name of the field that contains the embedding
                    "k": 10 #How many data entries to retrieve
                }
            }
        }
    ])
    similar_docs = list(similar_docs_cusor)

    return similar_docs
    

#Set OpenAI API key
openai.api_key = "OPENAI KEY"

#Get user input
input_search_string = input("Enter the search string: ")

#Generate an openai embedding from the user input
input_embedding = get_embedding(input_search_string)
print("embedding generated, length of embedding: {}".format(len(input_embedding)))


#Connect to MongoDB. 
# NOTE: Make sure the connection string, db_name and collection_name are correct)
client = pymongo.MongoClient("MONGODB CONNECTION STRING")

db_name = "sample_airbnb"
collection_name = "listingsAndReviews"
collection = client[db_name][collection_name]

#Perform Vector Search
output_data = findSimilarDocuments(collection,input_embedding)

#Print some fields of the result data
for val in output_data:
    print("Name: {}, summary: {}, bedrooms {}.\n".format(val.get("name"),val.get("summary"),val.get("bedrooms")))


