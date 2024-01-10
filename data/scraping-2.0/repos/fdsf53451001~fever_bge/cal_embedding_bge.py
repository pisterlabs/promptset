from langchain.embeddings import HuggingFaceBgeEmbeddings
import numpy as np

model_name = 'BAAI/bge-large-en-v1.5'
model_kwargs = {'device':'cuda:1'}
encode_kwargs = {'normalize_embeddings':True}
model = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs,
    # query_instruction="Represent this sentence for searching relevant passages:"
)

def get_embeddings():
    return model

def calculate_query_embedding(query):
    return model.embed_query(query)

def calculate_docs_embedding(docs):
    return model.embed_documents(docs)

def calculate_cosine_sim(a, b):
    # Compute the dot product of a and b
    dot_product = np.dot(a, b)
    # Compute the L2 norm of a and b
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # Compute the cosine similarity
    sim = dot_product / (norm_a * norm_b)
    return sim

# text="Trump’s Misleading Claims About Electric Vehicles and the Auto Industry"
# query_vector = calculate_query_embedding(text)

# texts = ['Trump said President Joe Biden “has dictated that nearly 70% of all cars” made in the U.S. must be “fully electric” in 10 years. The administration cannot mandate how many cars must be all-electric. It proposed new emission standards, and how the industry meets the new rules is up to them',
#          'During a campaign swing through South Carolina on Sept. 25, former President Donald Trump stopped by a boat factory, spoke to supporters at a rally, and took a tour of a gun store.',
#          'Former U.S. Ambassador to the United Nations Nikki Haley accused Florida Gov. Ron DeSantis of banning fracking and offshore drilling in his state. While DeSantis has supported such bans, he hasn’t actually implemented them.',
#          'A Trump spokesperson, Steven Cheung, posted on X on Sept. 25 that the former president bought the gun during his visit to the store.',
#          ]
# text_vectors = calculate_docs_embedding(texts)

# for i, text_vector in enumerate(text_vectors):
#     similarity = calculate_cosine_sim(query_vector, text_vector)
#     print(i, similarity)