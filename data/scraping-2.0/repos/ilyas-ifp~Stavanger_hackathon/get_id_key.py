from dotenv import load_dotenv
import os
import json
from openai import AzureOpenAI
from langchain.vectorstores import FAISS
import glob
load_dotenv(".env.shared")
load_dotenv(".env.secret")

openai_client = AzureOpenAI(
    api_version=os.environ["OPENAI_API_VERSION"],
    azure_endpoint=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"]
)
# Path to your JSON file
# file_path = 'result_4.json'

# # Sets to store unique keys and indices
# unique_keys = set()
# unique_indices = set()

# # Load and parse the JSON data
# with open(file_path, 'r') as file:
#     data = json.load(file)

#     # Iterate through the items
#     for item in data:
#         unique_keys.add(item['key'])
#         unique_indices.add(item['index'])

# # Now unique_keys and unique_indices contain only unique values
# print("Unique Keys:", unique_keys)
# print("Unique Indices:", unique_indices)

db = FAISS.load_local("data/embeddings",openai_client.embeddings.create(model=os.environ["ADA002_DEPLOYMENT"],input=""))
    
# docs = db.similarity_search_by_vector(query[0])


# # Save the map to a JSON file
# with open('index_key_map.json', 'w') as file:
#     json.dump(index_key_map, file, indent=4)

print("Map saved in 'index_key_map.json'")


def main() :
    json_files = glob.glob(os.path.join('json/updated', '*.json'))

    # Process each JSON file
    for file_path in json_files:
        unique_keys = set()
        unique_indices = set()
        # Load the JSON data
        with open(file_path, 'r') as file:
            data = json.load(file)

            # Iterate through the items
        for item in data:
            unique_keys.add(item['key'])
            unique_indices.add(item['index'])
        
        # Initialize an empty map
        index_key_map = {}

        dict = db.docstore._dict
        list_dict = list(dict.items())
        for e in unique_indices :
            doc = [key for idx, key in enumerate(list_dict) if idx == e][0]
            index_key_map[e] = doc[0]
            # print(doc)

        # Update the JSON data
        for item in data:
            if item['index'] in index_key_map:
                item['index'] = index_key_map[item['index']]

        # Write the updated data back to a JSON file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        print("ok")


if __name__ == '__main__' :
    main()