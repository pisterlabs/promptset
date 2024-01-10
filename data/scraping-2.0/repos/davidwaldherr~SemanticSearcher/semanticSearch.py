import pandas as pd
import numpy as np
import openai
openai.api_key ='INSERT API KEY HERE'

df = pd.read_csv('Summarize/bookHereSentences.csv')
df['babbage_search'] = df.babbage_search.apply(eval).apply(np.array)

from openai.embeddings_utils import get_embedding, cosine_similarity

# # prompt the user if they would like to embed all of the sentences in the book
def summarizeDecision():
    summary = input('Enter text to sort: ')
    return summary

mySummary = summarizeDecision()

# search through the reviews for a dream sentence
def search_reviews(df, dreamsign_description, pprint=True):
    embedding = get_embedding(dreamsign_description, engine='text-search-babbage-query-001')
    df['similarities'] = df.babbage_search.apply(lambda x: cosine_similarity(x, embedding))

    res = df.sort_values('similarities', ascending=False)
    if pprint:
        for r in res:
            print(r[:200])
            print()
    return res
res = search_reviews(df, mySummary)
# iterate and print res.iloc[i]['excerpt'] to a text file
with open('Summarize/bookHereSemanticResults.txt', 'w') as f:
    for i in range(30):
        f.write(str(res.iloc[i]['sentence']))
        f.write(str(res.iloc[i]['similarities']))
        f.write('\n')