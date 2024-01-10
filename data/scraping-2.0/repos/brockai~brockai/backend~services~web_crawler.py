import os
import pandas as pd
import tiktoken
import openai
import numpy as np
from ast import literal_eval
from helpers.domain import crawl, answer_question
from helpers.processing import remove_newlines, split_into_many, max_tokens

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

# Define OpenAI api_key
# openai.api_key = '<Your API Key>'

# Define root domain to crawl
domain = "openai.com"
full_url = "https://openai.com/"

crawl(full_url)

texts=[]

# Get all the text files
for file in os.listdir("text/" + domain + "/"):

    with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
        text = f.read()

        # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
        texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

df = pd.DataFrame(texts, columns = ['fname', 'text'])

# Set the text column
df['text'] = df.fname + ". " + remove_newlines(df.text)
df.to_csv('processed/scraped.csv')
df.head()

tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']

df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution histogram
# df.n_tokens.hist()    

shortened = []

# Loop through the dataframe to tune tokens
for row in df.iterrows():

    if row[1]['text'] is None:
        continue

    # if n_tokens > Max_tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])
    else:
        shortened.append( row[1]['text'] )

df = pd.DataFrame(shortened, columns = ['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# df.n_tokens.hist()

# Note that you may run into rate limit issues: https://platform.openai.com/docs/guides/rate-limits

df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
df.to_csv('processed/embeddings.csv')
df.head()

df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)

df.head()

print(answer_question(df, question="What day is it?", debug=False))

print(answer_question(df, question="What is our newest embeddings model?"))
