import os, types
import openai
from sentence_transformers import SentenceTransformer
import tiktoken
import chromadb
from chromadb.config import Settings

chroma_client = chromadb.PersistentClient(path="/tmp/cdb")
collection = chroma_client.get_collection("Books_Index001") # ST Embeddings
#collection = chroma_client.get_collection("Books_Index002") # OAI Embeddings
embeddingsmodel = SentenceTransformer('all-MiniLM-L6-v2')

OPENAIKEY= os.environ.get('OPENAIKEY')
openai.api_key = OPENAIKEY

oaimodel = "gpt-3.5-turbo"
oai_model_token_limit = 1000000000
oai_encoding = tiktoken.encoding_for_model(oaimodel)
indexEliminator = 0

def oai_embedding_function(data):
    #print(embeddingsData)
    response = openai.Embedding.create(
    input=data,
    model="text-embedding-ada-002")
    #print(response)
    embeddings = list(map(lambda row: row.embedding,response.data))
    return embeddings

def embedding_function(data):
   embeddings = embeddingsmodel.encode(data)
   return embeddings.tolist()

def system_prompt_template(context):
    return (f"Answer the question only using the context provided in 'Context:' section.\n\n"
          + f"Context:\n\n"
          + f"{context}:\n\n")

def collateMatches(m):
    series = ""
    description = ""
    try:
        if m['services'] != None:
            series = m['services']
    except Exception as e:
        pass
    try:
         if m['description'] != None:
            description = m['description']
    except Exception as e:
        pass
    c_ = "title:"+m['title']+";\n author:"+m['author']+";\n rating:"+str(m['rating'])+";\n description:"+ description
    #c_ = "title:"+m['title']+";\n author:"+m['author']+";\n rating:"+str(m['rating'])
    return c_

while True:
    user_input = input("Enter your query: ")
    if user_input == 'Q' or user_input == 'q':
        break
    queryEmbedding = embedding_function(user_input)
    #queryEmbedding = oai_embedding_function(user_input)
    query_response = collection.query(
        query_embeddings=queryEmbedding,
        n_results=10
    )
  
    distances = query_response['distances'][0]
    metadatas = query_response['metadatas'][0]
    
    print(distances)
    
    matchedVectors = list((map(lambda m: collateMatches(m),
                                   metadatas[0:len(metadatas)-indexEliminator])));  
    while True:
        #print(distances[0:len(metadatas)-indexEliminator])
        usableVectors = matchedVectors[0:len(metadatas)-indexEliminator]
        matchedVectorsCount = len(usableVectors)
        augmentedQuery =   "\n\n\n".join(usableVectors)
        prompt = system_prompt_template(augmentedQuery).join(user_input)        
        prompt_token_count = len(oai_encoding.encode(prompt))
        if prompt_token_count <= oai_model_token_limit:
            break
        print(f"With {matchedVectorsCount} matches, Prompt Length {prompt_token_count} Exceed model limit {oai_model_token_limit}, reducing context length by eliminating elements with largest distance in matched vector")
        indexEliminator+=1
    
    try:
            completion = openai.ChatCompletion.create(
                        model = oaimodel,
                        messages = [
                                    {"role": "system", "content": system_prompt_template(augmentedQuery)},
                                    {"role": "user", "content": user_input}
                                ],
                        temperature=0.1
                            )
    except:
            print("Error Occured")
    else:
        print("")
        print(completion.choices[0].message.content)
        print("")
        print("[Tokens Consumed "+str(completion.usage.total_tokens)+"]")
        print("")
        indexEliminator = 0