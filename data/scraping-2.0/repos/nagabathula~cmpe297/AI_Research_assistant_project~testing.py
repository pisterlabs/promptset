import os
import pinecone
import validators
import json
from collections import defaultdict
from openai import OpenAI

# Define all necessary functions

def fetch_abstract(url):

    r = session.get(url)
    content = r.html.find("#content-inner", first=True)
    abstract = content.find(".abstract", first=True).text
    return abstract

def avg_score(papers):
    avg_score = sum([p.score for p in papers]) / len(papers)
    return round(avg_score, 2)

def get_embedding(text, model,client):
    print("inside get_embedding")
    #check for clinet object
    try :
        print(client)
        embedding = client.embeddings.create(input = [text], model=model).data[0].embedding
    except Exception as e:
        print("Encountered error when fetching embedding from OpenAI: {e}", flush=True)
        return error("OpenAI not responding. Try again in a few minutes.")
    print ("after embedding call")
    print(embedding)
    
    return embedding
     

def get_matches(index, k, vector=None, id=None, exclude=None):
    assert vector is not None or id is not None
    if vector is not None:
        top_k = index.query(vector=vector, top_k=k, include_metadata=True)
    else:
        top_k = index.query(id=id, top_k=k, include_metadata=True)
    matches = top_k["matches"]
    papers = [Paper(match) for match in matches if match["id"] != exclude]
    authors = get_authors(papers)
    
    papers = [paper.__dict__ for paper in papers[:10]]
    return json.dumps({"papers": papers, "authors": authors})

def get_authors(papers):
    authors = defaultdict(list)
    for paper in papers:
        for author in paper.authors_parsed:
            authors[author].append(paper)
    authors = [{"author": author,
                "papers": [paper.__dict__ for paper in papers],
                "avg_score": avg_score(papers)}
                for author, papers in authors.items()]
    authors = sorted(authors, key=lambda e: e["avg_score"], reverse=True)
    authors = sorted(authors, key=lambda e: len(e["papers"]), reverse=True)
    return authors[:10]

def error(msg):
    return json.dumps({"error": msg})

class Paper(dict):
    def __init__(self, match):
        super().__init__()
        
        self.id = match["id"]
        self.score = round(match["score"], 2)
        
        metadata = match["metadata"]
        self.title = metadata["title"]
        self.authors = metadata["authors"]
        self.abstract = metadata["abstract"]
        self.year = metadata["year"]
        self.month = metadata["month"]
        
        authors_parsed = self.authors.split(",")
        self.authors_parsed = [author.strip() for author in authors_parsed]

def search(query):
    
    openai_api_key = "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"
    pinecone_api_key = "afb4b293-66c3-4f26-9db9-6f211f21112d"

    if not openai_api_key or not pinecone_api_key:
        raise ValueError("API keys for OpenAI and Pinecone are required.")
    
    client = OpenAI(api_key=openai_api_key)
    pinecone.init(api_key=pinecone_api_key, environment="asia-southeast1-gcp-free")
    index_name = "arvix-index-1"
    MODEL = "text-embedding-ada-002"
    index = pinecone.Index(index_name)
    
    ##query = "machine learning"
    K = 2 # number of matches to request from Pinecone
    
    # special logic for handling arxiv url queries
    if validators.url(query):
        arxiv_id = query.split("/")[-1]
        matches = index.fetch([arxiv_id])["vectors"]
        if len(matches) == 0:
            abstract = fetch_abstract(query)
            embed = get_embedding(abstract, MODEL)
            return get_matches(index, K, vector=embed, exclude=arxiv_id)
        return get_matches(index, K, id=arxiv_id, exclude=arxiv_id)
    
    # reject natural language queries longer than 200 characters
    if len(query) > 200:
        return error("Sorry! The length of your query cannot exceed 200 characters.")
    
    # embed query using OpenAI API
    try:
        print(query)
        print(MODEL)
        print("Fetching embedding from OpenAI...")
        embed = get_embedding(query, MODEL,client)
        print(embed);
    except Exception as e:
        print("Encountered error when fetching embedding from OpenAI: {e}", flush=True)
        return error("OpenAI not responding. Try again in a few minutes.")
    
    # once we have the query embedding, find closest matches in Pinecone
    try:
        print("Fetching matches from Pinecone...")
        matches = get_matches(index, K, vector=embed)
        return matches
    except Exception as e:
        print(f"Encountered error when fetching matches from Pinecone: {e}", flush=True)
        return error("Pinecone not responding. Try again in a few minutes.")

# Main execution block
if __name__ == "__main__":
    openai_api_key = "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"
    pinecone_api_key = "afb4b293-66c3-4f26-9db9-6f211f21112d"

    if not openai_api_key or not pinecone_api_key:
        raise ValueError("API keys for OpenAI and Pinecone are required.")

    client = OpenAI(api_key=openai_api_key)
    pinecone.init(api_key=pinecone_api_key, environment="asia-southeast1-gcp-free")
    index_name = "arvix-index-1"
    MODEL = "text-embedding-ada-002"
    index = pinecone.Index(index_name)
    matches = search()
    # Convert the dictionary to a beautified JSON string
    formatted_json = json.dumps(matches, indent=4)

    # Print the beautified JSON
    print(formatted_json)   
    
