from openai import OpenAI
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset


ss_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
client = OpenAI()

df = pd.read_csv('final_train_dataset.tsv', sep='\t')
dataset = Dataset.from_pandas(df)

idioms_train = [x for i,x in enumerate(dataset['Idiom']) if i%2]
meanings = [x for i,x in enumerate(dataset['Meaning']) if i%2]
explanations = [x for i,x in enumerate(dataset['Explanation']) if i%2]
langs = ['spanish','german']

english_meaning_embeds = [ss_model.encode(x) for x in meanings]

n = 8

kmeans = KMeans(n_clusters=n, random_state=42, n_init=10).fit(english_meaning_embeds)

cluster_labels = kmeans.labels_
centers = kmeans.cluster_centers_
representatives = []

for x in centers:
    best = 0
    rep = english_meaning_embeds[0]
    curr_best_dist = np.abs(np.linalg.norm(x-rep))
    for i,y in enumerate(english_meaning_embeds):
        distance = np.abs(np.linalg.norm(x-y))
        if distance < curr_best_dist:
            curr_best_dist = distance
            best = i
    representatives.append(idioms_train[best])

idiom_clusters = [[] for _ in range(n)]
for i in range(len(idioms_train)):
    idiom_clusters[cluster_labels[i]].append(idioms_train[i])


df = pd.read_csv('final_test_dataset.tsv', sep='\t')
dataset2 = Dataset.from_pandas(df)

idioms_test = [x for i,x in enumerate(dataset2['Idiom']) if i%2]

with open('cluster_labels.txt', 'w') as file:
    responses = []
    for i,x in enumerate(idioms_test):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a clustering expert."},
                {"role": "user", "content": f"Which of the following is '{x}' most similar to? 1. {representatives[0]}, 2. {representatives[1]}, 3. {representatives[2]}, 4. {representatives[3]}, 5. {representatives[4]}, 6. {representatives[5]}, 7. {representatives[6]}, 8. {representatives[7]}? Respond only with your choice."}
            ],
            temperature=0.2
        )
        file.write(response.choices[0].message.content + '\n')
        file.flush()
        print(i)


top_five = [[] for _ in range(n)]

for i, x in enumerate(centers):
    dists = [np.abs(np.linalg.norm(x-y)) for y in english_meaning_embeds]
    indices = np.argsort(dists)[:5]
    top_five[i] = [idioms_train[z] for z in indices]

with open('cluster_insights.txt', 'w') as file: 
    for x in top_five:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert on idiomatic expressions."},
                {"role": "user", "content": f"What is an overarching theme of the following group of idiomatic expressions? 1. {x[0]}, 2. {x[1]}, 3. {x[2]}, 4. {x[3]}, 5. {x[4]}"}
            ],
            temperature=0.2
        )
        file.write(str(x) + '\n')
        file.write(response.choices[0].message.content.replace('\n', ' ') + '\n\n')
        file.flush()


# any: 'Explain the idiomatic expression {ie}'
# single: 'Explain in one {lang} sentence the idiomatic expression {ie}'
# two or less: 'Explain in no more than two {lang} sentences the meaning of the idiomatic expression {ie}'

with open('explanation_responses_two_max.txt', 'w') as file:
    for i,x in enumerate(idioms_test):
        for y in langs:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f"Explain in no more than two {y} sentences the meaning of the idiomatic expression '{x}'"}
                ],
                temperature=0.5
            )
            file.write(response.choices[0].message.content.replace('\n', ' ') + '\n')
            file.flush()
            print(i)


with open('cluster_explanation_responses_two_max.txt', 'w') as outfile:
    with open('cluster_labels.txt', 'r') as infile:
        labels = [line[:-1] for line in infile]
        for i,x in enumerate(idioms_test):
            for y in langs:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": f"Based on its similarity to {labels[i]}, explain in no more than two {y} sentences the meaning of the idiomatic expression '{x}'"}
                    ],
                    temperature=0.2
                )
                outfile.write(response.choices[0].message.content.replace('\n', ' ') + '\n')
                outfile.flush()
                print(i)


with open('pipeline_explanation_responses_two_max.txt', 'w') as file:
    for i,x in enumerate(idioms_test):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Explain in no more than two sentences the meaning of the idiomatic expression '{x}'"}
            ],
            temperature=0.5
        )
        ex = response.choices[0].message.content.replace('\n', ' ')
        for y in langs:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f"Translate the following into {y}: '{ex}'"}
                ],
                temperature=0.5
            )
            file.write(response.choices[0].message.content.replace('\n', ' ') + '\n')
            file.flush()
            print(i)
