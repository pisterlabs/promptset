# Sparse Dense and Hybrid Search

## Remove old Weaviate DB files

!rm -rf ~/.local/share/weaviate

## Recreate the example
##
"""
With the same data as in the previous lesson

So, dense search uses vector embeddings
representation of the data to perform the search. So, it
relies on the meaning of the data in order
to perform that query. So for example, if we look for baby dogs
maybe we can get back information and content
on puppies. However, this has its limitations. For example,
if the model that we are using was trained on
a completely different domain, the accuracy
of our queries would be rather poor. It's very much
like if you went to a doctor and asked them how to
fix a car engine. Well, the doctor probably wouldn't have a good
answer for you.
Another example is when we're dealing with stuff like serial numbers,
like seemingly random strings of text. And
in this case, also, there isn't a lot of meaning into codes like BB43300, right?
Like if you ask the semantic engine for finding content
with that, you will get high quality results back. This is
why we need to actually go into a different
direction for situations like this and try to go
for keyword search, also known as sparse search. Sparse search
is a way that allows you to utilize
the keyword matching across all of your content. One example
could be, hey, we could use bag of words. And the
idea behind bag of words is like for every passage
of text that you have in your data, what you
can do is grab all the words and then keep adding and
expanding to your table of available words, just like
you see below.
So in this case, we can see that like maybe extremely,
and cute appeared once in this sentence,
and then word eat appears twice. So, that's how we can construct
that for sparse embedding for this object. And
as I mentioned, this is called sparse embedding
because if you have across all of your data, are
so many words, actually, the vector that will represent
that data will have a lot of slots where you could count
each individual word. But in reality, you
would be catching maybe 1% of available words. So, you'd have
a lot of zeros in your data. A good example of a keyword-based algorithm is
Best Matching 25, also known as BM25. And it actually
performs really well when it comes to searching
across many, many keywords. And the idea behind it is that
it counts the number of words within the phrase that you are
passing in and then those that appear more
often are weighted as like less important when
the match occurs but words that are rare
if we match on that the score is a lot higher. And like you
see this example here the sentence that we
provided at the bottom will result in quite
a lot of zeros that's why we call it sparse vector search.
"""

import requests
import json

# Download the data
resp = requests.get('https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')
data = json.loads(resp.text)  # Load data

# Parse the JSON and preview it
print(type(data), len(data))

def json_print(data):
    print(json.dumps(data, indent=2))

import weaviate, os
from weaviate import EmbeddedOptions
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

client = weaviate.Client(
    embedded_options=EmbeddedOptions(),
    additional_headers={
        "X-OpenAI-Api-BaseURL": os.environ['OPENAI_API_BASE'],
        "X-OpenAI-Api-Key": openai.api_key,  # Replace this with your actual key
    }
)
print(f"Client created? {client.is_ready()}")

# Uncomment the following two lines if you want to run this block for a second time.
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

## Queries

### Dense Search

response = (
    client.query
    .get("Question", ["question", "answer"])
    .with_near_text({"concepts":["animal"]})
    .with_limit(3)
    .do()
)

json_print(response)

### Sparse Search - BM25

response = (
    client.query
    .get("Question",["question","answer"])
    .with_bm25(query="animal")
    .with_limit(3)
    .do()
)

json_print(response)

### Hybrid Search

response = (
    client.query
    .get("Question",["question","answer"])
    .with_hybrid(query="animal", alpha=0.5)
    .with_limit(3)
    .do()
)

json_print(response)

response = (
    client.query
    .get("Question",["question","answer"])
    .with_hybrid(query="animal", alpha=0)
    .with_limit(3)
    .do()
)

json_print(response)

response = (
    client.query
    .get("Question",["question","answer"])
    .with_hybrid(query="animal", alpha=1)
    .with_limit(3)
    .do()
)

json_print(response)
