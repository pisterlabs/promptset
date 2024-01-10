import cohere
import pandas as pd

# Load your CSV file
df = pd.read_csv('deposition.csv')

# Initialize Cohere client
api_key = 'COHERE_KEY'
co = cohere.Client(api_key)


# Function to generate embeddings
def generate_embeddings(text):
    return co.embed(model='small', texts=[text]).embeddings[0]


# Apply the function to the 'IndexText' column
df['Embeddings'] = df['IndexText'].apply(generate_embeddings)

# Saving the DataFrame with embeddings to a new CSV file
df.to_csv('deposition_embedding.csv', index=False)
