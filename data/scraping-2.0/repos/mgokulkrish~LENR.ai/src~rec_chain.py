from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from huggingface_hub import login
login("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")

import pickle

persist_directory = 'faiss_db/'
embedding_model = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
vector_db = FAISS.load_local(persist_directory, embeddings)


question = "What factors controls the ability of palladium cathods to attain high loading levels?"
result_docs = vector_db.similarity_search(question, k=10)

candidates = {}
for i, doc in enumerate(result_docs):
    md = (doc.metadata)
    candidates[md['doc_id']] = 1

with open('abstracts.pkl', 'rb') as file:
    abstracts = pickle.load(file)

candidate_abstracts = ""
for i in candidates:
    candidate_abstracts += f"doc{i}.pdf: {abstracts[i-1][0:300]};"

prompt = f""" [Inst] You are a recommendation system which recommends papers based on search. Given below
are the following. 1. The prompt from user 2. The candidate papers and its incomplete abstract seperated by semicolon.
Give recommendation of top 3 papers and insights about these candidates for future research.
User prompt: {question} 
Candidates: {candidate_abstracts} 
[/INST]
"""
print(prompt)