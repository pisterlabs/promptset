import pinecone
import openai
import os
import pprint
from rich import print

pp = pprint.PrettyPrinter(indent=2)

openai.api_key = os.environ['OPENAI_API_KEY']
embed_model = "text-embedding-ada-002"

index_name = os.environ['PINECONE_INDEX']
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],  # app.pinecone.io (console)
    environment=os.environ['PINECONE_ENVIRONMENT']  # next to API key in console
)
index = pinecone.GRPCIndex(index_name)

query = input("Enter your question to be augmented: ")
res = openai.Embedding.create(
    input=[query],
    engine=embed_model
)

# retrieve from Pinecone
xq = res['data'][0]['embedding']

# get relevant contexts (including the questions)
res = index.query(xq, top_k=5, include_metadata=True)

pp.pprint(res)

contexts = [item['metadata']['text'] for item in res['matches']]

augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query

# system message to 'prime' the model
primer = f"""you are an experienced frontend developer who cares deeply about code readability. A highly intelligent developer that answers
user questions based on the information provided by the user above
each question. If the information can not be found in the information
provided by the user you truthfully say "I don't know".
"""

res = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)
     
print(res["choices"][0]["message"]["content"])