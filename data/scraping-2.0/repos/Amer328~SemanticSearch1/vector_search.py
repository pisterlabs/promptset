import pinecone
from sentence_transformers import SentenceTransformer,util
import streamlit as st

model = SentenceTransformer('all-MiniLM-L6-v2') #384 dimensional

pinecone_api_key = st.secrets["PINECONE_API"]

pinecone.init(api_key=pinecone_api_key, environment="us-east-1-aws") 

index = pinecone.Index("research3")

# view index stats
index.describe_index_stats()

import cohere
cohere_api_key = st.secrets["COHERE_API"]
# os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY") or getpass.getpass()
# init client
# co = cohere.Client(os.environ["COHERE_API"])
co = cohere.Client(cohere_api_key)

def rebuildIndex():
    pinecone.delete_index('research3')
    pinecone.create_index('research3', dimension=384,metric='cosine', replicas=1, pod_type='s1.x1')


def get_docs(query: str, top_k: int):
  # encode query
  xq = model.encode([query]).tolist()
  
  # search pinecone index
  res = index.query(xq, top_k=top_k, include_metadata=True)

  # get doc text and title
  docs = {x['metadata']['context']: {'index': i, 'title': x['metadata']['title']} for i, x in enumerate(res["matches"])}
  return docs


def addData(corpusData,url):
    id  = index.describe_index_stats()['total_vector_count']
    # strip out the path leading up to the filename
    last_slash_idx = url.rfind('\\')
    filename = url[last_slash_idx+1:] 
    
    for i in range(len(corpusData)):
        chunk=corpusData[i]
        chunkInfo=(str(id+i),
                model.encode(chunk).tolist(),
                {'title': filename,'context': chunk})
        index.upsert(vectors=[chunkInfo])


def find_match(query, k):
    docs = get_docs(query, top_k=k)
    rerank_docs = co.rerank(
        query=query, documents=docs.keys(), top_n=k, model="rerank-english-v2.0")

    results = []  # List to store the results

    for doc in rerank_docs[:6]:
        document_text = doc.document["text"]
        document_info = docs.get(document_text)

        if document_info:
            title = document_info['title']
            # Append the title and document as a tuple to the results list
            results.append((title, document_text))

    return results  # Return the accumulated results
