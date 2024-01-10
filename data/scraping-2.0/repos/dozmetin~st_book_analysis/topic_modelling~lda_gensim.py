import os

from gensim import corpora, models
from gensim.models import CoherenceModel
from preprocessing import preprocess_text_list
import pdfplumber
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import numpy as np

# The path of the data (chapters)
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def read_pdf_directory(directory):
    """
    Reads the pdf files and turns them into string.
    :param directory: The data directory
    :return: A string as a result of reading the files
    """
    corpus = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):  # More file extensions can be added if preferred.
            filepath = os.path.join(directory, filename)
            with pdfplumber.open(filepath) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text()
                corpus.append(text)
    return corpus


# Our corpus as a string
corpus = read_pdf_directory(data_dir)

# Preprocess the corpus
prepared_corpus = preprocess_text_list(corpus)

# Tokenize the corpus
tokenized_corpus = [w.split() for w in corpus]

# Dictionary from the tokenized corpus
dictionary = corpora.Dictionary(tokenized_corpus)

# Bag-of-words representation of the corpus
bow_corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_corpus]

# Our LDA model (topic number is selected to be 13)
# Alpha and beta parameters are set to be auto. The model automatically learns the best values.
lda_model = models.LdaModel(
    corpus=bow_corpus,
    id2word=dictionary,
    num_topics=13,
    passes=1000,
    chunksize=1000,
    random_state=100,
    alpha='auto',
    eta='auto',
)

# Print the top words for each topic
for topic_id, topic in lda_model.print_topics():
    print(f"Topic {topic_id}: {topic}")


if __name__ == "__main__":

    # Create the pyLDAvis visualization html file
    vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary=lda_model.id2word)
    pyLDAvis.save_html(vis, 'pyLDAvis_visualization_results.html')

    # Calculate the CV coherence score
    cv_coherence = CoherenceModel(model=lda_model, corpus=bow_corpus, dictionary=dictionary, coherence='c_v')
    cv_score = cv_coherence.get_coherence()
    print('CV Coherence Score: ', cv_score)

# The below method is used for topic number parameter optimization.
# The coherence score per topic number iteration will be stored.
def compute_coherence_values(dictionary, corpus, bow, limit, start, step):
    """
    Calculates CV coherence for various number of topics
    :param dictionary: The dictionary
    :param corpus: The corpus
    :param bow: Bag of words representation of the corpus
    :param limit: The maximum number of topics
    :param start: Topic number to start the iteration
    :param step: The amount to increase the topic number in each iteration
    :return: The list of coherence values
    """
    coherence_values = []
    for num_topic in np.arange(start, limit, step):
        lda_model2 = models.LdaModel(
            corpus=bow_corpus,
            id2word=dictionary,
            num_topics=num_topic,
            passes=1000,
            random_state=100,
            alpha='auto',
            eta='auto',
        )
        coherencemodel = CoherenceModel(model=lda_model2, corpus=bow_corpus, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return coherence_values


# Uncomment the below code for topic number optimization plotting.

# coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, bow=bow_corpus, start=10, limit=30, step=1)
# # Plot the results
# limit=30; start=10; step=1
# x = np.arange(start, limit, step)
# plt.plot(x, coherence_values)
# plt.xlabel("Topic Number")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.show()
