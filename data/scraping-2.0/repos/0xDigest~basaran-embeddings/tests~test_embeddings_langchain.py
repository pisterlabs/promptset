from langchain.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer, util

OPENAI_API_KEY = "none" 
OPENAI_API_BASE = "http://localhost:8088/v1"

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
    deployment="sentence-transformers",
    model="text-embedding-ada-0023",
)

sentence = "once upon a time,"
sentence2 = "once, in a time long ago,"

# compare local sentence transformers with model
em1 = embeddings.embed_query(sentence)
em2 = embeddings.embed_query(sentence2)
cos_score = util.cos_sim(em1, em2).tolist()[0][0]
print("API Similarity:", cos_score)

st_em1 = model.encode(sentence)
st_em2 = model.encode(sentence2)
cos_score = util.cos_sim(st_em1, st_em2).tolist()[0][0]
print("SentenceTranformers Similarity:", cos_score)

#API Similarity: 0.6838658452033997
#SentenceTranformers Similarity: 0.6842294931411743
