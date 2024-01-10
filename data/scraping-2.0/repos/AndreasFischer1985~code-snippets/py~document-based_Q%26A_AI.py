#############################################################################
# Title: Document-based Q&A example with sentence-embeddings & FOSS-AI
# Author: Andreas Fischer
# First published: March 20, 2023 
# last update: June 26, 2023
#############################################################################

from transformers import pipeline
from sentence_transformers import util
from sentence_transformers import SentenceTransformer 
import numpy as np
import requests

# Phase 1: generate sentence embeddings for each chunk of information (sentence)
#-------------------------------------------------------------------------------

sentences=["The meaning of life is to love","The meaning of vacation is to relax","Roses are red.","Hack the planet!"] # sentences to embed

def embedding(sentences,model):
  if("sentence-transformers" in model or "all-MiniLM" in model or "LaBSE" in model or "roberta-sentence" in model or "STS" in model):
    bot = SentenceTransformer(model)
    embeddings=bot.encode(sentences)
  elif("instructor" in model):  
    from langchain.embeddings import HuggingFaceInstructEmbeddings
    bot = HuggingFaceInstructEmbeddings(
      model_name=model,
      embed_instruction="Represent the document for retrieval: ",
      query_instruction="Represent the query for retrieval: " 
    )
    embeddings=bot.embed_documents(sentences) 
  else:
    bot = pipeline(task="feature-extraction", model=model)
    embeddings=bot(sentences)
    embeddings=np.concatenate(embeddings,0)
  return embeddings

if(False):
  embedding_model="sentence-transformers/all-MiniLM-L6-v2" # English sentence embeddings
  embedding_model="aari1995/German_semantic_STS_V2" # German sentence embeddings
  embedding_model="T-Systems-onsite/cross-en-de-roberta-sentence-transformer" # English & German sentence embeddings
  embedding_model="sentence-transformers/LaBSE" # Multilingual sentence embeddings
  embedding_model="hkunlp/instructor-xl" # English sentence embeddings 

embedding_model="hkunlp/instructor-large" # English sentence embeddings 
  
def normalize(v):
    norm = np.linalg.norm(v)
    if norm <= 1E-8: 
       return v
    return list(v / norm)

embeddings=embedding(sentences,embedding_model) # calculate embeddings
embeddings=np.array([normalize(embeddings[i]) for i in range(0,embeddings.shape[0])]) # normalize embeddings so that np.linalg.norm(v)==1
np.savetxt("embeddings.txt", embeddings, fmt = '%g', delimiter = '\t') # save embeddings for future usage

if(False): # Demo: for each sentence print similarity to first sentence
  for i in range(len(sentences))[1:]: # for loop to compare every sentence to first sentence
    if(i==1): print(sentences[0]) # print first sentence at the beginning
    cosine=util.pytorch_cos_sim(embeddings[0],embeddings[i]) # calculate cosine-similarity for sentence i
    print(sentences[i]+"; sim:"+str(cosine.numpy()[0][0])) # print result


# Phase 2: generate sentence embedding for a query and refine answer based on initial answer and most relevant chunk 
#--------------------------------------------------------------------------------------------------------------------

if(False):
  model="google/flan-t5-large"
  model="google/flan-t5-xl"  
  model="akosal/LongForm-T5-xl"	
  model="declare-lab/flan-alpaca-large"
  model="declare-lab/flan-alpaca-xl"
  model="google/flan-ul2"      
  model="stabilityai/stablelm-tuned-alpha-3b"
  model="writer/camel-5b-hf"
  model="tiiuae/falcon-7b"
  model="tiiuae/falcon-7b-instruct"
  
model="tiiuae/falcon-7b-instruct"

question="What is the meaning of life?"
arr = np.loadtxt("embeddings.txt", delimiter = '\t')        # load sentence-embeddings
emb = embedding(question,embedding_model)                   # encode new query
emb = np.array(normalize(emb))                              # normalize embeddings so that np.linalg.norm(v)==1
sim = np.dot(emb, arr[:, :].T).flatten()                    # compute cosine similarity   
ids = np.argsort(sim)[::-1]                                 # sort similarities in decreasing order
dev = [sentences[i]+" sim:"+str(sim[i]) for i in ids][:10]  # return up to 10 most similar texts
res = [sentences[i] for i in ids][:10]                      # return up to 10 most similar texts
print(res)

components={
  "query_str":question,
  "existing_answer":"Nothing.",
  "context_msg":"The meaning of life is to love.",
  "system":"You are a helpful and clever chatbot."}

refine_tmpl=("The original question is as follows: {query_str}\n"
  "We have provided an existing answer: {existing_answer}\n"
  "We have the opportunity to refine the existing answer "
  "(only if needed) with some more context below.\n"
  "------------\n"
  "{context_msg}\n"
  "------------\n"
  "Given the new context, refine the original answer to better "
  "answer the question. "
  "If the context isn't useful, return the original answer.")
  
def query(payload,model_id="google/flan-ul2",params={"max_length":200, "num_beams":16, "early_stopping":True}): 
  url = f"https://api-inference.huggingface.co/models/{model_id}"
  post = requests.post(url, json={"inputs":payload, "parameters":params})
  print(post)
  print(post.json())
  return post.json()[0]["generated_text"]


print(question)
a1=query(question,model) # Initial query
print(a1) # Flan-UL2's response: to be happy
components['question']=question
components['existing_answer']=a1 
components['context_msg']=res[0] 
refined_question=refine_tmpl.format(**components)
print(refined_question)
a2=query(refined_question,model) # Query refined response (with relevant context)
print(a2) # Flan-UL2's response: The meaning of life is to love 

components['context_msg']=res[-1] 
refined_question=refine_tmpl.format(**components)
print(refined_question)
a2b=query(refined_question,model) # Query refined response (with irrelevant context) 
print(a2b) # Flan-UL2's response: to be happy

