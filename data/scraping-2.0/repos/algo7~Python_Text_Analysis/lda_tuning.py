# Libraries
from gensim.models import CoherenceModel, LdaMulticore
from gensim.utils import ClippedCorpus
import tqdm
import numpy as np
import pandas as pd

"""
This module contains functions for tuning hyperparameters of an LDA model.
"""


def compute_coherence_values(corpus, dictionary, text, k, a, b):
    """
    Compute the coherence score of an LDA model.

    This function creates an LDA (Latent Dirichlet Allocation) model using 
    the gensim LdaMulticore class and then computes the coherence score of the 
    model using the CoherenceModel class, also from gensim.

    Parameters
    ----------
    corpus : iterable of list of tuple
        A collection of documents in the gensim corpus format, which is a 
        list of documents with each document being a list of (word_id, word_frequency) tuples.
    dictionary : gensim.corpora.dictionary.Dictionary
        A gensim Dictionary object, mapping between words and their integer ids.
    text : list of list of str
        The tokenized text data used for computing coherence values.
        Example: [['token1', 'token2'], ['token3', 'token4']]
    k : int
        The number of topics to be extracted by the LDA model.
    a : (float, 'symmetric', 'asymmetric')
        The alpha parameter of the LDA model, representing per-document topic distributions.
        If a is set to 'symmetric' or 'asymmetric', the respective type of prior will be used.
    b : (float, 'symmetric')
        The beta parameter of the LDA model, representing per-topic word distributions.
        If b is set to 'symmetric', a symmetric prior will be used.

    Returns
    -------
    float
        Coherence score of the LDA model, which is a single float value.

    Notes
    -----
    The function uses the LdaMulticore model, which is suitable for multicore processors,
    and calculates the coherence score with the 'c_v' method.

    Example
    -------
    >>> from gensim.corpora.dictionary import Dictionary
    >>> texts = [['sample', 'text'], ['another', 'text']]
    >>> dictionary = Dictionary(texts)
    >>> corpus = [dictionary.doc2bow(text) for text in texts]
    >>> compute_coherence_values(corpus, dictionary, texts, 2, 'symmetric', 'symmetric')
    """
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=dictionary,
                             num_topics=k,
                             random_state=100,
                             chunksize=100,
                             passes=10,
                             alpha=a,
                             eta=b)

    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=text, dictionary=dictionary, coherence='c_v')

    return coherence_model_lda.get_coherence()


def tune(corpus, dictionary, alpha_start, alpha_step, eta_start, eta_step, min_topics=2, max_topics=5, validation_set_percentage=0.75):
    """
    Perform a grid search to tune hyperparameters of an LDA model.

    This function exhaustively explores the hyperparameter space defined by specified
    ranges of the number of topics, alpha, and beta (eta) parameters for LDA modeling.
    It iteratively trains LDA models and calculates their coherence scores using different
    hyperparameter combinations and saves the results to a CSV file for further analysis.

    Parameters
    ----------
    corpus : iterable of list of tuple
        A collection of documents in the gensim corpus format, which is a list of documents
        with each document being a list of (word_id, word_frequency) tuples.
    dictionary : gensim.corpora.dictionary.Dictionary
        A gensim Dictionary object, mapping between words and their integer ids.
    alpha_start : float
        The starting value of the alpha parameter.
    alpha_step : float
        The step size of the alpha parameter.
    eta_start : float
        The starting value of the beta parameter.
    eta_step : float
        The step size of the beta parameter.
    min_topics : int, optional
        The minimum number of topics to be explored. The default is 2.
    max_topics : int, optional
        The maximum number of topics to be explored. The default is 5.
    validation_set_percentage : float, optional
        The percentage of the corpus to be used as a validation set. The default is 0.75.

    Notes
    -----
    The function considers the following hyperparameter ranges:
    - Number of topics (k): [min_topics, max_topics) with a step size of 1.
    - Alpha parameter (a): [alpha_start, 1) with a step size of alpha_step, plus 'symmetric' or 'asymmetric'.
    - Beta parameter (eta): [eta_start, 1) with a step size of eta_step, and 'symmetric'.

    Two validation sets, validation_set_percentage% of the corpus and the full corpus, are used to assess coherence.

    The resulting dataframe containing the coherence scores and the hyperparameter values
    used for each iteration is written to a CSV file named 'lda_tuning_results.csv'.

    Coherence values are calculated using the `compute_coherence_values` function (not defined
    in this snippet). Ensure this function is defined in your script and it accepts parameters 
    (corpus, dictionary, k, a, e) where 'e' is the beta parameter.

    WARNING: This function can be computationally expensive and take a long time to run,
    especially on larger corpora or with a wide range of hyperparameters.

    Example
    -------
    >>> from gensim.corpora.dictionary import Dictionary
    >>> texts = [['sample', 'text'], ['another', 'text']]
    >>> dictionary = Dictionary(texts)
    >>> corpus = [dictionary.doc2bow(text) for text in texts]
    >>> tune(corpus, dictionary)
    """
    grid = {}

    grid['Validation_Set'] = {}

    # Create a range of numbers of topics to try
    topics_range = range(min_topics, max_topics, 1)

    # Alpha parameter
    # Create a list of alpha parameters with the given start and step size
    alpha = list(np.arange(alpha_start, 1, alpha_step))
    # Add 'symmetric' and 'asymmetric' to the list of alpha parameters regardless of the step size and start value
    alpha.append('symmetric')
    alpha.append('asymmetric')

    # Beta parameter
    eta = list(np.arange(eta_start, 1, eta_step))
    # Add 'symmetric' to the list of beta parameters regardless of the step size and start value
    eta.append('symmetric')

    # Validation sets
    num_of_docs = len(corpus)
    corpus_sets = [ClippedCorpus(corpus, int(num_of_docs*validation_set_percentage)),
                   corpus]

    corpus_title = [f'{validation_set_percentage} Corpus', '100% Corpus']

    model_results = {'Validation_Set': [],
                     'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': []
                     }

    # Can take a long time to run
    if 1 == 1:
        # Set progress bar length
        pbar = tqdm.tqdm(total=(len(eta)*len(alpha) *
                                len(topics_range)*len(corpus_title)))

        # iterate through validation corpuses
        for i in range(len(corpus_sets)):
            # iterate through number of topics
            for k in topics_range:
                print(k)
                # iterate through alpha values
                for a in alpha:
                    # iterare through beta values
                    for e in eta:
                        # get the coherence score for the given parameters
                        cv = compute_coherence_values(corpus=corpus_sets[i],
                                                      dictionary=dictionary,
                                                      k=k, a=a, e=e)
                        # Save the model results
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(e)
                        model_results['Coherence'].append(cv)
                        # Update progress bar by 1
                        pbar.update(1)
        # Write the results to a csv file
        pd.DataFrame(model_results).to_csv(
            './lda_tuning_results.csv', index=False)

        # Close the progress bar
        pbar.close()


# Read the fine tuning results
results = pd.read_csv('./lda_tuning_results.csv')

# Plot coherence vs number of topics
results.groupby('Topics').max()['Coherence'].plot()

# Plot coherence vs alpha
results.groupby('Alpha').max()['Coherence'].plot()

# Plot coherence vs beta
results.groupby('Beta').max()['Coherence'].plot()
