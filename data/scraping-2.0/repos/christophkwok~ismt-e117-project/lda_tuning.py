import pandas as pd
import gensim
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

from recommender import general_preprocessing, generate_lda_dependencies


df = general_preprocessing(pd.read_csv("data/books.csv"))


id2word, corpus, data_ready = generate_lda_dependencies(df)

# Compute Coherence Score (the higher, the better)
### finding the optimal number of topics for LDA
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    num_topics_list = []

    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(
            corpus=corpus, num_topics=num_topics, id2word=id2word
        )
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence="c_v"
        )
        coherence_values.append(coherencemodel.get_coherence())
        num_topics_list.append(num_topics)

    return model_list, coherence_values, num_topics_list


# needs to be ordered to have plot display correctly
# or some sorting implemented (ordering this list was easier)
experiment = [
    # limit, start, step
    (20, 2, 2),
    (500, 50, 50),
    (1000, 500, 100),  # chris: changed start from 100 to 500
    (2000, 1000, 500),  # chris: changed start from 500 to 1000
]

model_list = []
coherence_values = []
num_topics_list = []

for limit, start, step in experiment:
    print(
        "computing coherence values for: limit={}, start={}, step={}".format(
            limit, start, step
        )
    )
    _model_list, _coherence_values, _num_topics_list = compute_coherence_values(
        dictionary=id2word,
        corpus=corpus,
        texts=data_ready,
        start=start,
        limit=limit,
        step=step,
    )
    model_list.extend(_model_list)
    coherence_values.extend(_coherence_values)
    num_topics_list.extend(_num_topics_list)

# Show graph for entire experiment
plt.plot(num_topics_list, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc="best")
plt.show()
