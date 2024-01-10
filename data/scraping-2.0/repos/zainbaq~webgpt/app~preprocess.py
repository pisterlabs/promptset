import os
import argparse
import tiktoken
import pandas as pd
import openai
from tqdm import tqdm
from utils import get_domain, get_openai_key


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', '-u', required=True, help='url to scrape')
    parser.add_argument('--domain', '-d', help='domain to restrict scrape over')
    parser.add_argument('--max_tokens', default=500, help='max_tokens for embedding vectors')
    return parser.parse_args()

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

args = parse_args()

domain = args.domain # <- put your domain to be crawled
# Get domain if not provided
if not domain:
    domain = get_domain(args.url)

full_url = args.url # <- put your domain to be crawled with https or http

# Create a list to store the text files
texts=[]

# Get all the text files in the text directory
for file in os.listdir("data/text/" + domain + "/"):

    # Open the file and read the text
    with open("data/text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
        text = f.read()

        # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
        texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

# Create a dataframe from the list of texts
df = pd.DataFrame(texts, columns = ['fname', 'text'])

# Set the text column to be the raw text with the newlines removed
df['text'] = df.fname + ". " + remove_newlines(df.text)
df.to_csv('data/processed/' + domain + '/scraped.csv')
# print(df.head())

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('data/processed/' + domain + '/scraped.csv', index_col=0)
df.columns = ['title', 'text']

# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution of the number of tokens per row using a histogram
# df.n_tokens.hist()

max_tokens = args.max_tokens

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

    return chunks


shortened = []

# Loop through the dataframe
for row in df.iterrows():

    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])

    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append( row[1]['text'] )

df = pd.DataFrame(shortened, columns = ['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
# df.n_tokens.hist()

openai.api_key = get_openai_key()
tqdm.pandas()

print('Creating embeddings...')
df['embeddings'] = df.text.progress_apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

if not os.path.exists(f"data/processed/{domain}"):
    os.mkdir(f"data/processed/{domain}")

df.to_csv('data/processed/' + domain + '/embeddings.csv')
df.head()