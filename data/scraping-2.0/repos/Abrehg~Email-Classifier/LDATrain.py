import pandas as pd
from text_parsing import normalize_text
import gensim.corpora as corpora
import gensim
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from gensim.models.coherencemodel import CoherenceModel

#data collection
data = pd.read_csv("LDA_Data.csv")
data = data.drop(columns=['Id', 'Has_Emoji', 'Class', 'Sender', 'Unnamed: 0'], axis=1)
data = data.drop(labels=[376, 385, 625, 881, 1104, 1344, 2223, 2428, 2467, 2723, 2916, 3077, 3254, 3423, 3573, 3731, 3869, 3921, 3997, 4004, 4152, 4335, 4505, 4661, 4803, 4934, 5091, 5206, 5207, 5208, 5285, 5438, 5686, 5779, 8034, 8171, 8846, 9368, 10191, 10192, 10193, 12067, 12089, 13374, 14136, 14433, 14506, 15017, 15195, 15210, 15221, 15283, 15299, 15316, 15550, 15551, 15552, 15553, 15680], axis=0)

indexes = [376, 385, 625, 881, 1104, 1344, 2223, 2428, 2467, 2723, 2916, 3077, 3254, 3423, 3573, 3731, 3869, 3921, 3997, 4004, 4152, 4335, 4505, 4661, 4803, 4934, 5091, 5206, 5207, 5208, 5285, 5438, 5686, 5779, 8034, 8171, 8846, 9368, 10191, 10192, 10193, 12067, 12089, 13374, 14136, 14433, 14506, 15017, 15195, 15210, 15221, 15283, 15299, 15316, 15550, 15551, 15552, 15553, 15680]
emoji = 0
S = 0
output = ""
input = ""
InputData = []
while S < len(data):
    for i in indexes:
        if S == i:
          S = S + 1
    input = data.at[S, "Subject"]
    output, emoji = normalize_text(input)
    InputData.append(output)
    S = S + 1

B = 0
Body = []
while B < len(data):
    for i in indexes:
        if B == i:
          B = B + 1
    input = data.at[B, "Body"]
    output, emoji = normalize_text(input)
    InputData.append(output)
    B = B + 1

# Create Dictionary
Dictionary = corpora.Dictionary(InputData)

# Create Corpus
texts = InputData

# Term Document Frequency
corpus = [Dictionary.doc2bow(text) for text in texts]

# number of topics
num_topics = 20

# Build LDA model
lda_model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=Dictionary,
                                       num_topics=num_topics,
                                       alpha = 'auto',
                                       random_state = 42,
                                       passes = 200)


# Print the Keyword in the 10 topics
doc_lda = lda_model[corpus]
# Save model to disk. (modify it to save as a separate file instead of a path)
lda_model.save("LDAModel.pickle")

"""
#Run this to find the optimal number of topics for model
print("optimization process started")
num_keywords = 15
num_topics = list(range(21)[1:])

print("models to be created")
LDA_models = {}
LDA_topics = {}
coherences = []
for i in num_topics:
    # problem creating models
    LDA_models[i] = gensim.models.LdaModel(corpus=corpus,
                             id2word=Dictionary,
                             num_topics=i,
                             passes=20,
                             alpha='auto',
                             random_state=42)
    print(f"Model number {i} has been created")
    shown_topics = LDA_models[i].show_topics(num_topics=i,
                                             num_words=num_keywords,
                                             formatted=False)
    LDA_topics[i] = [[word[0] for word in topic[1]] for topic in shown_topics]
    print("Topics listed")



coherences = [CoherenceModel(model=LDA_models[i], corpus=corpus, dictionary=Dictionary, coherence='u_mass').get_coherence()\
              for i in num_topics[:-1]]
print("coherence data added to list")
print("models created")

def jaccard_similarity(topic_1, topic_2):
    #Derives the Jaccard similarity of two topics
    #Jaccard similarity:
    #- A statistic used for comparing the similarity and diversity of sample sets
    #- J(A,B) = (A ∩ B)/(A ∪ B)
    #- Goal is low Jaccard scores for coverage of the diverse elements
    intersection = set(topic_1).intersection(set(topic_2))
    union = set(topic_1).union(set(topic_2))

    return float(len(intersection)) / float(len(union))
print("similarity function created")

LDA_stability = {}
for i in range(0, len(num_topics) - 1):
    jaccard_sims = []
    for t1, topic1 in enumerate(LDA_topics[num_topics[i]]):  # pylint: disable=unused-variable
        sims = []
        for t2, topic2 in enumerate(LDA_topics[num_topics[i + 1]]):  # pylint: disable=unused-variable
            sims.append(jaccard_similarity(topic1, topic2))

        jaccard_sims.append(sims)

    LDA_stability[num_topics[i]] = jaccard_sims
print("stability done")

mean_stabilities = [np.array(LDA_stability[i]).mean() for i in num_topics[:-1]]
print("mean stabilities done")

print(f"coherences data: {coherences}")
print(f"mean stabilities data: {mean_stabilities}")
coh_sta_diffs = [coherences[i] - mean_stabilities[i] for i in range(num_keywords)[:-1]] # limit topic numbers to the number of keywords
coh_sta_max = max(coh_sta_diffs)
coh_sta_max_idxs = [i for i, j in enumerate(coh_sta_diffs) if j == coh_sta_max]
ideal_topic_num_index = coh_sta_max_idxs[0] # choose less topics in case there's more than one max
ideal_topic_num = num_topics[ideal_topic_num_index]

plt.figure(figsize=(20, 10))
ax = sns.lineplot(x=num_topics[:-1], y=mean_stabilities, label='Average Topic Overlap')
ax = sns.lineplot(x=num_topics[:-1], y=coherences, label='Topic Coherence')

ax.axvline(x=ideal_topic_num, label='Ideal Number of Topics', color='black')
ax.axvspan(xmin=ideal_topic_num - 1, xmax=ideal_topic_num + 1, alpha=0.5, facecolor='grey')

y_max = max(max(mean_stabilities), max(coherences)) + (0.10 * max(max(mean_stabilities), max(coherences)))
ax.set_ylim([-10, y_max])
ax.set_xlim([1, num_topics[-1] - 1])

ax.axes.set_title('Model Metrics per Number of Topics', fontsize=25)
ax.set_ylabel('Metric Level', fontsize=20)
ax.set_xlabel('Number of Topics', fontsize=20)
plt.legend(fontsize=20)
plt.show()
"""