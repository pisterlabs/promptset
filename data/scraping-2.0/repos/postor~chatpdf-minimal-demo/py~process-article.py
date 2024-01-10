import numpy as np
import openai
import json
import os.path
import pprint
import hashlib
import sys
from pathlib import Path
import math
import pandas as pd
from sklearn.cluster import KMeans

COMPLETIONS_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
CONTEXT_TOKEN_LIMIT = 1500
TOKENS_PER_TOPIC = 2000
TOPIC_NUM_MIN = 3
TOPIC_NUM_MAX = 10


def get_topic_num(sources):
  num = math.floor(len("".join(sources))/TOKENS_PER_TOPIC)
  if num<TOPIC_NUM_MIN:
    return TOPIC_NUM_MIN
  if num>TOPIC_NUM_MAX: 
     return TOPIC_NUM_MAX
  return num

def get3questions(sources,embeddings):
  matrix = np.vstack(embeddings)
  print(np.shape(np.array(embeddings).tolist()))
  df = pd.DataFrame({"embedding":np.array(embeddings).tolist(),"p":sources})
  n_clusters = get_topic_num(sources)
  kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
  kmeans.fit(matrix)
  df["Cluster"] = kmeans.labels_  

  df2 = pd.DataFrame({"tokens":[],"prompts":[]})
  for i in range(n_clusters):
    ctx = u""
    ps = df[df.Cluster == i].p.values
    for x in ps:
      if len(ctx)>CONTEXT_TOKEN_LIMIT:
        continue
      ctx+= u"\n"+x
    prompt = u"Suggest a simple, clear, single, short question base on the context, answer in the same language of context\n\nContext:"+ctx+u"\n\nAnswer with the language used in context, the question is:"
    df2.loc[len(df2)] = [len("".join(ps)),prompt]
  
  questions = []
  for prompt in df2.sort_values('tokens',ascending=False).prompts.sample(3).values:
    print(prompt)
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content":prompt}])
    questions.append(completion.choices[0].message.content)
    print(completion.choices[0].message.content)
  return questions

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
  folder = 'embeddings/cache/'
  Path(folder).mkdir(parents=True, exist_ok=True)
  tmpfile = folder+hashlib.md5(text.encode('utf-8')).hexdigest()+".json"
  if os.path.isfile(tmpfile):
      with open(tmpfile , 'r', encoding='UTF-8') as f:
          return json.load(f)
  result = openai.Embedding.create(
    model=model,
    input=text
  )

  with open(tmpfile, 'w',encoding='utf-8') as handle2:
      json.dump(result["data"][0]["embedding"], handle2, ensure_ascii=False, indent=4)

  return result["data"][0]["embedding"]


def file2embedding(folder,contents=""):
  embeddings = []
  sources = []
  content = contents
  Path(folder).mkdir(parents=True, exist_ok=True)
  
  if content == "":
    with open(folder+'/source.txt', 'r', encoding='UTF-8') as handle1:
      content = handle1.read()

  for source in content.split('\n'):
    if source.strip() == '':
        continue
    embeddings.append(get_embedding(source))
    sources.append(source)

  questions = get3questions(sources,embeddings)
  with open(folder+'/result.json', 'w',encoding='utf-8') as handle2:
    json.dump({"sources":sources,"embeddings":embeddings,"questions":questions}, handle2, ensure_ascii=False, indent=4)

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, embeddings) -> list[(float, (str, str))]:
    #pprint.pprint("embeddings")
    #pprint.pprint(embeddings)
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in enumerate(embeddings)
    ], reverse=True, key=lambda x: x[0])
    
    return document_similarities

def ask(question:str,embeddings,sources):
    ordered_candidates = order_document_sections_by_query_similarity(question,embeddings)
    ctx = u""
    for candi in ordered_candidates:
        next = ctx + u"\n" + sources[candi[1]]
        if len(next)>CONTEXT_TOKEN_LIMIT:
            break
        ctx = next
    if len(ctx) == 0:
      return u""    
    
    prompt = u"".join([
        u"Answer the question base on the context, answer in the same language of question\n\n"
        u"Context:" + ctx + u"\n\n"
        u"Question:" + question + u"\n\n"
        u"Answer:"
                        ])

    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content":prompt}])
    return [prompt, completion.choices[0].message.content]




if sys.argv[1] == 'compile':
    folder = sys.argv[2]
    contents = "" if len(sys.argv)<4 else sys.argv[3]
    file2embedding(folder,contents)
else: # query
    folder = sys.argv[2]
    question = sys.argv[3]
    with open(folder+'/result.json', 'r', encoding='UTF-8') as f: 
        obj=json.load(f)
        [prompt,answer] = ask(question,obj["embeddings"],obj["sources"])
        print(json.dumps({
            "question":question,
            "prompt":prompt,
            "answer":answer
        }))