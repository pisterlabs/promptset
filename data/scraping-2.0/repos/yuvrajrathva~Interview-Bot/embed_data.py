import openai
import os
import csv
import glob
from dotenv import load_dotenv


load_dotenv()

text_array = []
api_key = os.environ.get('OPENAI_KEY')
openai.api_key = api_key
dir_path = os.path.join(os.getcwd(), 'documents')
dir_full_path = os.path.join(dir_path, '*.txt')
embeddings_filename = "embeddings.csv"

# Loop through all .txt files in the /training-data folder
for file in glob.glob(dir_full_path):
    # Read the data from each file and push to the array
    # The dump method is used to convert spacings into newline characters \n
    with open(file, 'r') as f:
        text = f.read().replace('\n', '')
        text_array.append(text)

# This array is used to store the embeddings
embedding_array = []

if api_key is None or api_key == "YOUR_OPENAI_KEY_HERE":
    print("Invalid API key")
    exit()

# Loop through each element of the array
for text in text_array:
    # Pass the text to the embeddings API which will return a vector and
    # store in the response variable.
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )

    # Extract the embedding from the response object
    embedding = response['data'][0]["embedding"]

    # Create a Python dictionary containing the vector and the original text
    embedding_dict = {'embedding': embedding, 'text': text}
    # Store the dictionary in a list.
    embedding_array.append(embedding_dict)

with open(embeddings_filename, 'w', newline='') as f:
    # This sets the headers
    fieldnames = ['embedding', 'text']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for obj in embedding_array:
        # The embedding vector will be stored as a string to avoid comma
        # separated issues between the values in the CSV
        writer.writerow({'embedding': str(obj['embedding']), 'text': obj['text']})

print("Embeddings saved to:", embeddings_filename)