import cohere
import numpy as np

co = cohere.Client("COHERE_API_KEY")

# get the embeddings
words = ["Red", "Blood", "Sea"]
(p1, p2, p3) = co.embed(words).embeddings

# compare them
def calculate_similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#print(calculate_similarity(p1, p2)) # 0.8974134906653974 - very similar!

print(calculate_similarity(p1, p3)) # -0.5648101116086706 - not similar!