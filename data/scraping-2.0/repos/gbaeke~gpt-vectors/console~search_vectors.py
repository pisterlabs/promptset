import os
import pinecone
import openai
import requests
from bs4 import BeautifulSoup
import tiktoken

def tokens_from_string(string, encoding_name):
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

# given an list of dictionaries with metadata, score, retrieve the item with the highest score and print metadata
def get_highest_score_url(items):
    highest_score = 0
    highest_score_item = None
    for item in items:
        if item["score"] > highest_score:
            highest_score = item["score"]
            highest_score_item = item

    return highest_score_item["metadata"]['url']

# get the Pinecone API key and environment
pinecone_api = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENVIRONMENT')

pinecone.init(api_key=pinecone_api, environment=pinecone_env)

# set index
index = pinecone.Index('blog-index')

while True:
    # set query
    your_query = input("\nWhat would you like to know? ")
    
    # vectorize your query with openai
    try:
        query_vector = openai.Embedding.create(
            input=your_query,
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]
    except Exception as e:
        print("Error calling OpenAI Embedding API: ", e)
        continue

    # search for the most similar vector in Pinecone
    search_response = index.query(
        top_k=5,
        vector=query_vector,
        include_metadata=True)

    

    # get url with highest score
    url = get_highest_score_url(search_response['matches'])

    # print url
    print("Highest score url: ", url)

    # get the content of the article
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    # get the article
    article = soup.find('div', {'class': 'entry-content'}).text

    try:
        # openai chatgpt with article as context
        # chat api is cheaper than gpt: 0.002 / 1000 tokens
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                { "role": "system", "content": f"Provide your answer based on this context: {article}. Do not answer beyond this context!" },
                { "role": "user", "content": your_query }
            ],
            temperature=0,
            max_tokens=200
        )

        print(f"\n{response.choices[0]['message']['content']}")
    except Exception as e:
        print(f"Error with OpenAI Completion: {e}")




