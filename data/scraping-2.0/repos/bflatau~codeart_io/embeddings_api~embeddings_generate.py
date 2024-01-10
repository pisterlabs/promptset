import os
from openai import OpenAI
import pandas as pd
import re
import time
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())
client = OpenAI(
  api_key = os.getenv("OPENAI_API_KEY"),
)

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ") #not sure if this is needed?
    return client.embeddings.create(input = [text], model=model).data[0].embedding



ALLOWED_REGEX = r'[a-zA-Z\$\*\%\&\@\!\?\#\'\,\" ]*'

def translateToSplitflapAlphabet(text):
    return text.replace("'", '').replace('"', '*').replace(',', '')

def wrap(text, rowLength):
    rows = []

    lines = text.split('\n')
    for line in lines:
        curRow = []
        curRowLength = 0
        words = line.split(' ')

        for word in words:
            if curRowLength + len(word) <= rowLength:
                curRowLength += len(word) + 1
                curRow.append(word)
            else:
                rows.append(curRow)
                curRow = [word]
                curRowLength = len(word) + 1
        rows.append(curRow)

    return list(map(lambda row: ' '.join(row), rows))

def isDisplayableInOnePage(text):
    characters_ok = (re.fullmatch(ALLOWED_REGEX, text) is not None)
    wrapped = wrap(translateToSplitflapAlphabet(text), 18)
    return characters_ok and len(wrapped) <= 6




df = pd.read_csv('./embeddings/combined_season1-37.tsv',  sep='\t') #read file
df = df[[ 'answer', 'question']] #name the columns we want
df = df.dropna() #remove all the columns not identified above
print(len(df))

# Filter out embeddings with characters we can't display
df = df[df.apply(lambda row: isDisplayableInOnePage(row['answer']) and isDisplayableInOnePage(row['question']), axis=1)]
print("FILTERED")
print(len(df))

df['combined'] = "Question: " + df.question.str.strip() + "; Answer: " + df.answer.str.strip() #make superstring?


# Select a subset of rows to process
df = df.iloc[20000:120000]
print("SELECTED SUBSET")
print(len(df))

"""
Scott's notes from iteratively generating these:
0:1000 -- embedded_120k_FILTERED_ada_1704443394422
1000:6000 -- embedded_120k_FILTERED_ada_1704447175265
6000:20000 -- embedded_120k_FILTERED_ada_1704450383665
20000:120000 -- embedded_120k_FILTERED_ada_1704494377631

Joined via:
head -n 1 embedded_120k_FILTERED_ada_1704443394422.csv > embedded_ada.csv
tail -n+2 embedded_120k_FILTERED_ada_1704443394422.csv >> embedded_ada.csv
tail -n+2 embedded_120k_FILTERED_ada_1704447175265.csv >> embedded_ada.csv
tail -n+2 embedded_120k_FILTERED_ada_1704450383665.csv >> embedded_ada.csv
tail -n+2 embedded_120k_FILTERED_ada_1704494377631.csv >> embedded_ada.csv
"""

def getTransform():
    i = 0
    def transform(value):
        nonlocal i

        # Print the item we're processing every 100th item
        if i % 100 == 0:
            print(i, value)
        i += 1
        try:
            return get_embedding(value)
        except Exception as e:
            print(e)
            return "ERR"
    return transform

input("Press enter to start making openai API calls...")
df['ada_embedding'] = df.combined.apply(getTransform())

df.to_csv(f'./embeddings/embedded_120k_FILTERED_ada_{int(time.time()*1000)}.csv', index=False)


print("FINISHED!")