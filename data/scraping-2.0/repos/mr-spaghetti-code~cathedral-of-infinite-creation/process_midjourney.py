import json
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

openai_api_key = "sk-..."

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

f = open('intermediate.json')

with_embeddings = json.load(f)

f.close() 

print("Loaded data...")

print("Computing embeddings:")

with_embeddings = []

for element in data:
    try:
        text = element["prompt"]
        print(f"Computing embedding for prompt: {text}")
        query_result = embeddings.embed_query(text)
        element["embedding"] = query_result
        with_embeddings.append(element)
    except:
        print(f"There was an error on element: {element}")

with open("intermediate.json", "w") as outfile:
    outfile.write(json.dumps(with_embeddings, indent=4))

with_embeddings = with_embeddings[:1000]

embedding_matrix = []

for element in with_embeddings:
    embedding_matrix.append(element["embedding"])

similarities = cosine_similarity(embedding_matrix)

print(f"Length of cosine similarity matrix: {len(similarities)}")

# Actual input_json
# input_json = {
#     "nodes" : [],
#     "links" : []
# }


input_json = {
    "nodes" : [],
    "links" : {}
}

# {
#     "nodes": [
#         {
#           "id": "id1",
#           "name": "name1",
#           "val": 1
#         },
#         {
#           "id": "id2",
#           "name": "name2",
#           "val": 10
#         },
#         ...
#     ],
#     "links": [
#         {
#             "source": "id1",
#             "target": "id2"
#         },
#         ...
#     ]
# }
# array([[1.        , 0.77911004, 0.81937526, 0.80392125, 0.7655019 ],
#        [0.77911004, 1.        , 0.79608392, 0.81086091, 0.75359267],
#        [0.81937526, 0.79608392, 1.        , 0.79420443, 0.79456192],
#        [0.80392125, 0.81086091, 0.79420443, 1.        , 0.7746613 ],
#        [0.7655019 , 0.75359267, 0.79456192, 0.7746613 , 1.        ]])

# for n, element in enumerate(with_embeddings):
#     print(f"Processing element: #{n}")
#     input_json["nodes"].append({
#         "id" : element["id"],
#         "img_url" : element["image_paths"][0],
#         "prompt" : element["prompt"]
#     })

#     similarity_m = similarities[n]
#     for i, s in enumerate(similarity_m[n+1:]):
#         if s > 0.8 and element["id"] != with_embeddings[i]["id"]:
#             input_json["links"].append({
#                 "source" : element["id"],
#                 "target" : with_embeddings[i]["id"],
#                 "strength": s
#             })


for n, element in enumerate(with_embeddings):
    print(f"Processing element: #{n}")
    input_json["nodes"].append({
        "id" : element["id"],
        "img_url" : element["image_paths"][0],
        "prompt" : element["prompt"]
    })
    input_json["links"][element["id"]] = []
    similarity_m = similarities[n]
    for i, s in enumerate(similarity_m[n+1:]):
        if s > 0.8 and element["id"] != with_embeddings[i]["id"]:
            input_json["links"][element["id"]].append([with_embeddings[i]["id"], s])
            # input_json["links"].append({
            #     "source" : element["id"],
            #     "target" : with_embeddings[i]["id"],
            #     "strength": s
            # })

json_object = json.dumps(input_json, indent=4)

with open("network_new_format_100.json", "w") as outfile:
    outfile.write(json_object)

    

