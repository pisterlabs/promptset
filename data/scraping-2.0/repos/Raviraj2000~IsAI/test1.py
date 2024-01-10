import cohere
import pandas as pd
text = []
with open('Ai.txt') as f:
    txt = f.read()
    text = txt.split('.')
    print(text[:-1])    
    f.close()
    
df_AI = []

for line in text[:-1]:
    df_AI.append([line, 'AI'])

with open('Human.txt' ,encoding='utf-8') as f:
    txt = f.read()
    text = txt.split('.')
    print(text)   
    f.close()

df_H = []

for line in text[:-1]:
    df_H.append([line, 'Human'])
    
    
df = df_AI + df_H
    
df = pd.DataFrame(df, columns = ['Text', 'Class'])

import cohere
from cohere.classify import Example
co = cohere.Client(API_KEY)

examples = []

for row in range(len(df)):
    examples.append(Example(df['Text'][row], df['Class'][row]))

text = 'i have a head that, whenever i take a peek in the mirror, looks like someone spit into it.'

classifications = co.classify(model='large', inputs=[text], examples=examples)

classifications.classifications[0].prediction

