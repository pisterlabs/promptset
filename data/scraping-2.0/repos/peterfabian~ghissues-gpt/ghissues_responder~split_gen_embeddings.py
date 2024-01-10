import os
import json
import pandas as pd
import tiktoken
import matplotlib
import openai
from sentence_transformers import SentenceTransformer
from config import *

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ', regex=False)
    serie = serie.str.replace('\\n', ' ', regex=False)
    serie = serie.str.replace('  ', ' ', regex=False)
    serie = serie.str.replace('  ', ' ', regex=False)
    return serie

def remove_boilerplate(serie):
    boilerplates = [
        'Bug Description:',
        'Is your feature request related to a problem? Please describe.',
        'Isolating the problem (mark completed items with an [x]):',
        'I have deactivated other plugins and confirmed this bug occurs when only WooCommerce plugin is active.',
        'This bug happens with a default WordPress theme active, or Storefront.',
        'I can reproduce this bug consistently using the steps above.',
        'I have searched for similar bugs in both open and closed issues and cannot find a duplicate. Describe the bug',
        'I have have carried out troubleshooting steps and I believe I have found a bug.',
        'Please provide us with the information requested in this bug report.'
        'Without these details, we won\'t be able to fully evaluate this issue.',
        'Bug reports lacking detail, or for any other reason than to report a bug, may be closed without action.',
        'Prerequisites (mark completed items with an [x]):',
        'Isolating the problem', 
        'This bug happens with only WooCommerce plugin active',
        'This bug happens with a default WordPress theme active, or Storefront',
        'I can reproduce this bug consistently using the steps above',
        'I have deactivated other plugins not needed to reproduce this bug and confirmed this bug occurs when only WooCommerce plugin is active.'
        'Prerequisites I have searched for similar issues in both open and closed tickets and cannot find a duplicate',
        'The issue still exists against the latest master branch of WooCommerce on Github (this is not the same version as on WordPress.org!)',
        'I have attempted to find the simplest possible steps to reproduce the issue',
        'I have included a failing test as a pull request (Optional)',

    ]

    for boilerplate in boilerplates:
        serie = serie.str.replace(boilerplate, ' ', regex=False)

    serie = serie.str.replace('This is a description of an issue with WooCommerce plugin. The author gave it the following title:', 'WooCommerce plugin issue with title: ', regex=False)
    serie = serie.str.replace('Log Directory Writable.*', ' ', regex=True)

    return serie


# Config
input_texts_path = "/Users/peter/Documents/GitHub/wc-org/gh-stats/"
file_name_pattern = "issues-2023-flattened-v3"
max_tokens = 500

# Create a list to store the text files
texts=[]

# Get all the text files in the text directory
for file in os.listdir(input_texts_path):
    if file_name_pattern in file:
        # Open the file and read the text
        with open(input_texts_path + "/" + file, "r") as f:
            issues = json.load(f)

            for issue in issues:
                # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
                texts.append((issue['url'], issue['content']))

# Create a dataframe from the list of texts
df = pd.DataFrame(texts, columns = ['url', 'text'])

# Set the text column to be the raw text with the newlines removed
df['text'] = remove_newlines(df.text)
df['text'] = remove_boilerplate(df.text)

df.to_csv('processed/issues-v3.csv')
# df.head()

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('processed/issues-v3.csv', index_col=0)
df.columns = ['url', 'text']

# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution of the number of tokens per row using a histogram
# df.n_tokens.hist()

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks

shortened = []

# Loop through the dataframe
for row in df.iterrows():

    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        chunks = split_into_many(row[1]['text'])
        for chunk in chunks:
            shortened.append((row[1]['url'], chunk))
    
    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append((row[1]['url'], row[1]['text']))

df_shortened = pd.DataFrame(shortened, columns = ['url', 'text'])
df_shortened['n_tokens'] = df_shortened.text.apply(lambda x: len(tokenizer.encode(x)))
# df_shortened.n_tokens.hist()

# this runs into a limit (currently 70/min)
# df_shortened['embeddings'] = df_shortened.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

# Use a different (local) model to get the embeddings. Using an assymetric model is better,
# see https://www.sbert.net/examples/applications/semantic-search/README.html#symmetric-vs-asymmetric-semantic-search
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(sentences=list(df_shortened['text']), convert_to_tensor=True)
df_shortened['embeddings'] = embeddings.tolist()

df_shortened.head()
df_shortened.to_csv('processed/embeddings.csv.gz', compression='gzip')