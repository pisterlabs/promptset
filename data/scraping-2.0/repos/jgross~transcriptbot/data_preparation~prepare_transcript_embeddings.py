import pandas as pd
import os
import openai
import csv
import pinecone
import string
import random
import json 
from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MODEL_NAME = "ada"
INDEX_NAME ='dotmaticsv2'

# DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
DOC_EMBEDDINGS_MODEL = "text-embedding-ada-002"


def get_number_of_words_in_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    # Split the text into a list of words
    words = text.split()
    return len(words)


def read_file_into_text_chunks(file_path, title, url):
    csv_output = {"text": [], "title": [], "url": []}
    # Initialize a variable to keep track of the current chunk
    current_chunk = ""

    # Initialize a variable to keep track of the word count
    word_count = 0

    # Check if the file exists
    if os.path.exists(file_path):
        number_of_words = get_number_of_words_in_file(file_path)
        # Open the file
        with open(file_path, "r") as f:
            total_word_count = 0
            # Iterate through each line in the file
            url = f.readline()
            date = f.readline()
            keywords = f.readline()
            for line in f:
                # Split the line into words
                words = line.split()
                
                # Iterate through each word in the line
                for word in words:
                    # Increment the word count
                    word_count += word.__len__()
                    total_word_count += 1
                    
                    # Add the word to the current chunk
                    current_chunk += word + " "
                    
                    # If the word count is 1,000, add the current chunk to the array and reset the current chunk and the word count
                    if word_count >= 400:
                        csv_output['text'].append(current_chunk)
                        csv_output['title'].append(title)
                        csv_output['url'].append(url)
                        current_chunk = ""
                        word_count = 0
            
            # If there are any remaining words in the current chunk after the loop is done, add them to the array
            if current_chunk:
                csv_output['text'].append(current_chunk)
                csv_output['title'].append(title)
                csv_output['url'].append(url)
            print(csv_output)
    else:
        # If the file does not exist, print an error message
        print("The file could not be found.")
    return(csv_output)


def get_embedding(text: str, model: str):
    print("Getting embedding for:" + text)
    try: 
        result = openai.Embedding.create(model=model, input=text)
        return result["data"][0]["embedding"]
    except Exception as e:
        print("Error: " + str(e))
        return ""

def write_json_to_file(json_obj, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(json_obj, outfile)

def read_directory_into_pinecone_embeddings(directory_path):
    # Initialize Pinecone with your API key and environment
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment="us-west1-gcp"
    )
    id = 0

    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(INDEX_NAME, dimension=1536)

    # Connect to the  index
    index = pinecone.Index(INDEX_NAME)

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename)) as file:
                print("Reading file: " + filename + "")
                title = filename.strip(".txt").strip(".mp3").replace("_", " ")
                print("Title: " + title)
                csv_output = read_file_into_text_chunks(os.path.join(directory_path, filename), title, INDEX_NAME)
                
                # Iterate through the text chunks
                for i in range(len(csv_output['text'])):
                    # Get the text chunk, url, and episode title for the current iteration
                    text = csv_output['text'][i]
                    url = csv_output['url'][i]
                    title = csv_output['title'][i]
                    id += 1

                    # Calculate the embedding for the text chunk using OpenAI
                    embedding = get_embedding(text, DOC_EMBEDDINGS_MODEL)

                    # Format the metadata in the desired format
                    meta = {'text': text, 'url': url, 'site': 'dotmatics.com', 'title': title}

                    # Save the embedding and meta data to the 'benchling' index in Pinecone
                    if index.upsert([(id.__str__(), embedding, meta)], namespace='dotmatics'):
                        write_json_to_file(embedding, os.path.join(directory_path, filename + '.json'))

                        


def main():
    read_directory_into_pinecone_embeddings("./dotmatics")
    
if __name__ == "__main__":
    main()