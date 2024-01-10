import gensim
import matplotlib.pyplot as plt
from gensim import corpora, models
from gensim.models import CoherenceModel

def compute_coherence_values(dictionary, corpus, texts, num_docs=0, start=2, limit=4, step=1, mallet_path=None):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        if mallet_path:
            model = gensim.models.wrappers.LdaMallet(mallet_path,
                                                     corpus=corpus,
                                                     workers=6,
                                                     id2word=dictionary,
                                                     random_seed=100,
                                                     num_topics=num_topics)
        else:
            model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=dictionary,
                                               workers=6,
                                               num_topics=num_topics,
                                               random_state=100,
                                               chunksize=num_docs,
                                               passes=10)
            
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model,
                                        texts=texts,
                                        dictionary=dictionary,
                                        coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def plot_c_v(x, coherence_values):
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.show()

def topics_in_doc(word_tokens, id2word, model):
    vec = id2word.doc2bow(word_tokens)
    return model[vec]