import utils

import pandas as pd
import matplotlib as plt

from gensim import corpora, models
from gensim.models import CoherenceModel
from scipy.sparse import csr_matrix


@utils.timer_decorator
def create_corpus(dtm: pd.DataFrame, dictionary: corpora.Dictionary, filename: str):
    corpus = []
    # Iterate over rows.
    for idx, row in dtm.iterrows():
        doc = []
        # Iterate over index.
        for seq, count in row.items():
            # Add each sequence to the document * count times.
            doc += [str(seq)] * int(count)
        corpus.append(dictionary.doc2bow(doc))

    # Write Corpus to disk.
    corpora.MmCorpus.serialize(filename, corpus)


@utils.timer_decorator
def create_corpus_(td_matrix: csr_matrix, dictionary: corpora.Dictionary, filename: str):
    corpus = []
    # Iterate over rows (documents).
    for idx in range(td_matrix.shape[0]):
        doc = [(seq_idx, int(count)) for seq_idx, count in enumerate(td_matrix[idx, :].toarray().flatten()) if
               count > 0]
        # Use the dictionary to convert document to bag of words format
        corpus.append(doc)

    # Write Corpus to disk.
    corpora.MmCorpus.serialize(filename, corpus)


@utils.timer_decorator
def create_dictionary(dtm: pd.DataFrame, filename: str):
    # Create a dictionary where the keys are sequences and the values are unique integer IDs.
    dictionary = corpora.Dictionary([list(map(str, dtm.columns))])

    # TODO -  Filter out words based on document frequency or other criteria
    # dictionary.filter_extremes(no_below=5, no_above=0.5)

    # Write Dictionary to disk.
    dictionary.save(filename)


@utils.timer_decorator
def create_dictionary_(seq_encoder, filename: str):
    # Create a dictionary where the keys are sequences and the values are unique integer IDs.
    dictionary = corpora.Dictionary([seq_encoder.classes_])

    # TODO -  Filter out words based on document frequency or other criteria
    # dictionary.filter_extremes(no_below=5, no_above=0.5)

    # Write Dictionary to disk.
    dictionary.save(filename)


@utils.timer_decorator
def train_model(corpus: corpora.MmCorpus, dictionary: corpora.Dictionary, filename: str = '', save: bool = True,
                num_topics: int = 10, random_state: int = 42, passes: int = 10, iterations: int = 200,
                chunksize: int = 20000, eval_every: int = 10):
    # Train LDA model on the corpus.
    model = models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, workers=7,
                                random_state=random_state,
                                # Number of passes through the corpus during training, higher is more accurate but slower.
                                passes=passes,
                                # Number of iterations over the corpus, controls how much model is updated.
                                iterations=iterations,
                                # Number of documents to be used in each training iteration, higher is faster but uses more memory.
                                chunksize=chunksize,
                                eval_every=eval_every)
    # Doesn't work with LdaMulticore.
    # alpha='auto', eta='auto')

    # Write model to disk.
    if save:
        model.save(filename)
    return model


@utils.timer_decorator
def optimize_parameters(dtm: pd.DataFrame, corpus: corpora.MmCorpus, dictionary: corpora.Dictionary,
                        min_topics: int = 5,
                        max_topics: int = 50, step_size: int = 5):
    # Range of number of topics to try.
    topics_range = range(min_topics, max_topics, step_size)

    # Transform tdm into list of lists, where each list contains the sequences in the document * count times.
    texts = [[str(seq)] * int(count) for idx, doc in dtm.iterrows() for seq, count in doc.items()]

    model_list = []
    coh_perp_values = []

    # Try each number of topics.
    for num_topics in topics_range:
        print("Training model with", num_topics, "topics.")
        model = train_model(corpus, dictionary, save=False, num_topics=num_topics)
        model_list.append(model)

        # Calculate perplexity score, the lower, the better.
        perplexity = model.log_perplexity(corpus)

        # Calculate coherence score, the higher, the better.
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coh_perp_values.append((coherence_model.get_coherence(), perplexity))
        print("Num Topics =", num_topics, " has Coherence Value of", coherence_model.get_coherence(),
              " and Perplexity Value of", perplexity)

    # Unzip coherence and perplexity values.
    coherence_values, perplexity_values = zip(*coh_perp_values)

    # Plotting.
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel("Num Topics")
    ax1.set_ylabel("Coherence score", color=color)
    ax1.plot(topics_range, coherence_values, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Perplexity score', color=color)
    ax2.plot(topics_range, perplexity_values, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()
