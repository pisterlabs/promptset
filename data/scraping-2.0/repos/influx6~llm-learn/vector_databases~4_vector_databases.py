## Vector Database setup

Remove old Weaviate DB files

!rm -rf ~/.local/share/weaviate


### Step 1 - Download sample data

import requests
import json

# Download the data
resp = requests.get('https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')
data = json.loads(resp.text)  # Load data

# Parse the JSON and preview it
print(type(data), len(data))

def json_print(data):
    print(json.dumps(data, indent=2))

json_print(data[0])

### Step 2 - Create an embedded instance of Weaviate vector database

import weaviate, os
from weaviate import EmbeddedOptions
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

client = weaviate.Client(
    embedded_options=EmbeddedOptions(),
    additional_headers={
        "X-OpenAI-BaseURL": os.environ['OPENAI_API_BASE'],
        "X-OpenAI-Api-Key": openai.api_key  # Replace this with your actual key
    }
)
print(f"Client created? {client.is_ready()}")

json_print(client.get_meta())

## Step 3 - Create Question collection

# resetting the schema. CAUTION: This will delete your collection
if client.schema.exists("Question"):
    client.schema.delete_class("Question")
class_obj = {
    "class": "Question",
    "vectorizer": "text2vec-openai",  # Use OpenAI as the vectorizer
    "moduleConfig": {
        "text2vec-openai": {
            "model": "ada",
            "modelVersion": "002",
            "type": "text",
            "baseURL": os.environ["OPENAI_API_BASE"]
        }
    }
}

client.schema.create_class(class_obj)

## Step 4 - Load sample data and generate vector embeddings

# reminder for the data structure
json_print(data[0])

with client.batch.configure(batch_size=5) as batch:
    for i, d in enumerate(data):  # Batch import data

        print(f"importing question: {i+1}")

        properties = {
            "answer": d["Answer"],
            "question": d["Question"],
            "category": d["Category"],
        }

        batch.add_data_object(
            data_object=properties,
            class_name="Question"
        )

count = client.query.aggregate("Question").with_meta_count().do()
json_print(count)

## Let's Extract the vector that represents each question!

# write a query to extract the vector for a question
result = (client.query
          .get("Question", ["category", "question", "answer"])
          .with_additional("vector")
          .with_limit(1)
          .do())

json_print(result)

## Query time
What is the distance between the `query`: `biology` and the returned objects?

response = (
    client.query
    .get("Question",["question","answer","category"])
    .with_near_text({"concepts": "biology"})
    .with_additional('distance')
    .with_limit(2)
    .do()
)

json_print(response)

response = (
    client.query
    .get("Question", ["question", "answer"])
    .with_near_text({"concepts": ["animals"]})
    .with_limit(10)
    .with_additional(["distance"])
    .do()
)

json_print(response)

## We can let the vector database know to remove results after a threshold distance!

response = (
    client.query
    .get("Question", ["question", "answer"])
    .with_near_text({"concepts": ["animals"], "distance": 0.24})
    .with_limit(10)
    .with_additional(["distance"])
    .do()
)

json_print(response)

## Vector Databases support for CRUD operations

### Create

#Create an object
object_uuid = client.data_object.create(
    data_object={
        'question':"Leonardo da Vinci was born in this country.",
        'answer': "Italy",
        'category': "Culture"
    },
    class_name="Question"
 )

print(object_uuid)

### Read

data_object = client.data_object.get_by_id(object_uuid, class_name="Question")
json_print(data_object)

data_object = client.data_object.get_by_id(
    object_uuid,
    class_name='Question',
    with_vector=True
)

json_print(data_object)

### Update

client.data_object.update(
    uuid=object_uuid,
    class_name="Question",
    data_object={
        'answer':"Florence, Italy"
    })

data_object = client.data_object.get_by_id(
    object_uuid,
    class_name='Question',
)

json_print(data_object)

### Delete

json_print(client.query.aggregate("Question").with_meta_count().do())

client.data_object.delete(uuid=object_uuid, class_name="Question")

json_print(client.query.aggregate("Question").with_meta_count().do())
