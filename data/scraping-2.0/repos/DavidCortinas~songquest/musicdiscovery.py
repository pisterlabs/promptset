import sys

print(sys.executable)
import openai
import pandas as pd
from sklearn.metrics.pairwise import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up OpenAI API credentials
openai.api_key = "YOUR_API_KEY"

# Load and preprocess data
data = pd.read_csv("data.csv")
data = data.dropna()
corpus = data["description"].tolist()

# Extract features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Compute pairwise similarities using cosine similarity
similarity_matrix = cosine_similarity(X)

# Generate recommendations based on user input
user_input = "Italian restaurant"
user_index = corpus.index(user_input)
recommendations = similarity_matrix[user_index].argsort()[:-6:-1]

# Print top 5 recommendations
print("Top 5 Recommendations:")
for i, index in enumerate(recommendations):
    print(
        f"{i+1}. {data.loc[index]['name']}: {data.loc[index]['description']}")
