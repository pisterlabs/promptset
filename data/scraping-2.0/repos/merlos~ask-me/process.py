import os
import pandas as pd
import tiktoken
import openai
import time
import argparse
import datetime

from config import CONFIG

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie


def create_corpus(text_files_path, corpus_csv):
    '''
    creates one single CSV file from a set of txt files stored in a folder 
    corpus a set of text files.
    It also does minimum processing calling `remove_newlines`
    
    text_files_path: string Path to text files
    corpus_csv: string path to the output file with the corpus
    ''' 
    # Create a list to store the text files
    texts=[]

    # Get all the text files in the text directory
    for file in os.listdir(text_files_path):
        print(f'   Processing {file}')
        # Open the file and read the text
        with open(text_files_path + file, "r", encoding="UTF-8") as f:
            text = f.read()
            texts.append((file, text))

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns = ['fname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv(corpus_csv)
    #df.head()

    print(f"Saved CSV file {corpus_csv}")

    
def tokenize_corpus(corpus_csv, max_tokens=500):
    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding('cl100k_base')

    df = pd.read_csv(corpus_csv, index_col=0)
    df.columns = ['title', 'text']

    
    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    shortened = []

    # Function to split the text into chunks of a maximum number of tokens
    def split_into_many(text, max_tokens = 500):

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


    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            chunks =  split_into_many(row[1]['text'], max_tokens)
            for chunk in chunks:
                shortened.append({'title': row[1].title, 'text': chunk})

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append({'title': row[1].title, 'text': row[1].text})

    df = pd.DataFrame(shortened, columns = ['title', 'text'])
    # Count tokens.
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
     
    return df

def process_embeddings(x):
    global current_embedding

    embedding = openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding']
    
    #print(x)
    current_embedding = current_embedding + 1 
    print(f'Processed embedding {current_embedding}/{number_of_embeddings}')
    # To prevent free account quota limit, sleep for some time
    time.sleep(CONFIG['idle_time']) 
    return embedding


# Instantiate the parser
parser = argparse.ArgumentParser(description='Process txt files into embeddings')
# Required positional argument
parser.add_argument('text_files_path', type=str, help='Folder with txt files to process. For example: ./text/www.merlos.org')
args = parser.parse_args()
text_files_path = args.text_files_path

# If input path does not exist -> stop
if not os.path.exists(text_files_path):
        parser.error(f"input_path directory '{text_files_path}' does not exist.")

# Now, it is time to get the embeddings of each chunk
# We will divide in batches, so, in case there is any issue with the session
# we can resume where we left

# define the batch size
batch_size = CONFIG['batch_size']

# check if there are any existing processed data files
processed_data_files = [f'./processed/{f}' for f in os.listdir('./processed/') if f.startswith('processed_batch_')]

df = None
start_index = 0
if processed_data_files:
    # Load the tokenized corpus
    df = pd.read_csv(CONFIG['tokenized_corpus_csv'], index_col=0)

    # if there are, find the last batch that was processed
    last_processed_batch = max([int(f.split('_')[2].split('.')[0]) for f in processed_data_files])
    start_index = (last_processed_batch + 1) * batch_size

else:
    # First, create corpus
    print(f'Text files in {text_files_path} will be processed to create the corpus')
    create_corpus(text_files_path, CONFIG['corpus_csv'])
    print(f"Corpus created. Saved in {CONFIG['corpus_csv']}")
    
    # Then tokenize corpus
    print (f'Corpus will be divided in chunks and tokenized...')
    df = tokenize_corpus(CONFIG['corpus_csv'], CONFIG['max_tokens'])
    df.to_csv(CONFIG['tokenized_corpus_csv'])
    print (f'Corpus successfully divided in chunks and tokenized.')

     # if not, start from the beginning of the dataframe
    start_index = 0

current_embedding = start_index
number_of_embeddings = df.shape[0]

print(f'The corpus has {df.shape[0]} total number of chunks of text.')
print(f'The corpus will be processed in batches of {batch_size}.')
if start_index != 0:
    print('*** Session recovered ***')
    print(f'Starting from {start_index}.')

# loop through the remaining part of the dataframe in batches
for i in range(start_index, len(df), batch_size):
    batch_df = df.iloc[i:i+batch_size].copy()
    #print(f'batch_df {batch_df.shape}')
    
    try:
        # process the batch dataframe 
        #processed_batch_df = batch_df.apply(lambda x: x**2)
        batch_df.loc[:, 'embeddings'] = batch_df.text.apply(process_embeddings)

        processed_batch_file = f'./processed/processed_batch_{i//batch_size}.csv'
        batch_df.to_csv(processed_batch_file, index=False)
        
        # append the processed data file name to the list
        processed_data_files.append(processed_batch_file)
        now = datetime.datetime.now()
        print(f'[{now.strftime("%H:%M")}]Saved batch {i//batch_size}')
    except Exception as e:
        print(f'Error processing batch {i//batch_size}: {e}')
        exit(1)
        break

# concatenate the processed data from all batches into a single dataframe
processed_dfs = []
for file in processed_data_files:
    processed_df = pd.read_csv(file)
    processed_dfs.append(processed_df)
processed_data = pd.concat(processed_dfs, ignore_index=True)

# save the concatenated processed dataframe to disk
processed_data.to_csv(CONFIG['embeddings_csv'], index=False)

# delete the temporary processed data files
for file in processed_data_files:
    os.remove(file)

print('Done.')
print('')
print('Now you can run ask.py to test the result')
print('    python ask.py this is my question')
print ('')
