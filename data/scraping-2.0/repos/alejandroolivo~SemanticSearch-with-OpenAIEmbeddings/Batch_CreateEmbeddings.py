import openai
import pandas as pd
import numpy as np
from getpass import getpass
from openai.embeddings_utils import get_embedding
import os
import configparser

# Read config
config = configparser.ConfigParser()
config.read('config.ini')

# Write KEY for openAI API
os.environ['OPENAI_API_KEY'] = config['openai']['api_key']

# Define the root directory path
rootdir = '.\Data\Tags Batches'

# Loop through each tag batch file
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith('.csv') and file.startswith('Tags_batch'):
            # Read the data from the csv file
            df = pd.read_csv(os.path.join(subdir, file))
            print(df)

            #Calculate word embeddings for each tag
            df['embedding'] = df['tag'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))

            # Export the word embeddings to a CSV file
            output_filename = os.path.join('.\Data\Word Embeddings Batches', file.replace('Tags', 'word_embeddings'))
            df.to_csv(output_filename, index=False)

            # Print the progress
            print('Word embeddings saved to ' + output_filename)

