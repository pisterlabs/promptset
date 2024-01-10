# Test Langchain Huggingfqace Embeddings + distances

from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np

# Testset
texts = [
    "king",
    "man",
    "woman",
    "queen",
    "horse",
    "car",
]

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name= 'all-MiniLM-L6-v2')

embeddings_for_text = {}
for text in texts:
    embeddings_for_text[text] = embeddings.embed_query(text)

for k,v in embeddings_for_text.items():
    print(f"'{k}', size=({len(v)}) ", end="")
print()

# distances (using Using dot() function)
def distance(a,b):
    t = np.subtract(a,b)
    return np.sqrt(np.dot(t.T, t))

print("king to queen", distance(embeddings_for_text["king"],embeddings_for_text["queen"]))
print()
print("king to man", distance(embeddings_for_text["king"],embeddings_for_text["man"]))
print("king to woman", distance(embeddings_for_text["king"],embeddings_for_text["woman"]))
print()
print("queen to woman", distance(embeddings_for_text["queen"],embeddings_for_text["woman"]))
print("queen to man", distance(embeddings_for_text["queen"],embeddings_for_text["man"]))
print()
print("man to woman", distance(embeddings_for_text["man"],embeddings_for_text["woman"]))
print()
print("king-man to queen-woman", distance(
    np.subtract(embeddings_for_text["king"],embeddings_for_text["man"]),
    np.subtract(embeddings_for_text["queen"],embeddings_for_text["woman"])
))
print()
print("king to horse", distance(embeddings_for_text["king"],embeddings_for_text["horse"]))
print("king to car", distance(embeddings_for_text["king"],embeddings_for_text["car"]))

      