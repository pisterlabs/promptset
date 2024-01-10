'''
a program for analyzing the topics within a selection of amazon reviews
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim_models

from help_functions import clean_data, get_doc_topic, generate_word_cloud


class LDATopicModel:
    '''
    class docstring
    '''

    def __init__(self, data_file='reviews_data.csv'):
        self.data_file = data_file


    def preprocess_data(self):
        '''
        preprocessing method to clean review text
        '''

        # load in dataset
        df = pd.read_csv(self.data_file)

        # create a clean_text column by applying clean_data to review text
        # (see helper functions for documentation)
        df['clean_text'] = df['reviews.text'].apply(clean_data)

        # create mask for docs in Electronics category to screen for this category
        electronics_mask = df.primaryCategories.isin(["Electronics"])
        self.df_electronics = df[electronics_mask]

        # load in the spaCy language model
        spacy.cli.download('en_core_web_sm')
        self.nlp = spacy.load('en_core_web_sm')


    def create_lemma_tokens(self):
        '''
        preprocessing method to lemmatize review text
        '''

        # create lemma tokens for each doc excluding stop words and punctuation
        # lemmatizing all this text takes a while
        print('Lemmatizing review text...')
        self.df_electronics['lemmas'] = self.df_electronics['clean_text'].apply(
            lambda x: [token.lemma_ for token in self.nlp(x) if \
                       (token.is_stop is False) and (token.is_punct is False)])


    def create_lemma_dictionary(self):
        '''
        dictionary creation method for generating mapping from token to numerical id
        '''

        # create lemma dictionary with Dictionary class
        self.id2word = corpora.Dictionary(self.df_electronics['lemmas'])

        # remove tokens appearning in less than three or more than half of docs
        self.id2word.filter_extremes(no_below=3, no_above=0.5)


    def generate_lda_models(self):
        '''
        model creation method which assembles corpus from bag-of-words model
        corpus used to generate LDA models with varying number of topics [2, 12]
        models and associated coherence values saved to lists for later comparison
        '''
                
        # create bag-of-words model and compute coherence values
        self.corpus = [self.id2word.doc2bow(lemma) for lemma in self.df_electronics['lemmas']]

        # calculate c_v coherence values
        print('Generating models and coherence scores...')

        # ordered list of models and associated coherence values for comparison
        self.coherence_values = []
        self.model_list = []

        # create models with varying number of topics from three to twelve
        for num_topics in range(3, 12):
            # create model with specific number of topics
            # save to list
            model = gensim.models.LdaModel(corpus=self.corpus,
                                        num_topics=num_topics,
                                        id2word=self.id2word,
                                        chunksize=100,
                                        passes=10,
                                        random_state=34,
                                        per_word_topics=True)
            self.model_list.append(model)

            # evaluate coherence score of model with specific number of topics
            # save to list at same index as model in above list
            coherence_model = CoherenceModel(model=model,
                                            texts=self.df_electronics['lemmas'],
                                            dictionary=self.id2word,
                                            coherence='c_v')

            self.coherence_values.append(coherence_model.get_coherence())

    def compare_models(self):
        '''
        comparison method for generated models
        graphs number of topics v. coherence score for each generated model
        saves model with highest coherence score as best_model attribute
        '''

        # graph coherence score v. no. topics to determine how many topics to generate
        plt.grid()
        plt.title("Coherence Score v. Number of Topics")
        plt.xticks(range(3, 12))
        plt.plot(range(3, 12), self.coherence_values, "-o")

        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")

        plt.show()

        # get model index with highest coherence score
        self.best_model = self.model_list[np.argmax(self.coherence_values)]
        # clearly five topics is the ideal number 


    def visualize_best_model(self):
        '''
        visualization method for best model
        NEEDS WORD CLOUD FOR EACH TOPIC TO BE CREATED
        creates pyLDAvis html file for interactive web visualization
        '''

        # prepare LDA model for visualization
        vis_data = pyLDAvis.gensim_models.prepare(self.best_model, self.corpus, self.id2word)
        # topics are roughly balanced that's interesting

        # create a dictionary to map topic IDs to meaningful names
        # I chose these names because I think they fit, however the choice is arbitrary in this case
        topic_name_dict = {0: 'use case', 1: 'price_value', 2: 'hardware', 3: 'aesthetics', 4: 'gift purchase'}

        # use get_topic_ids_for_docs to get topic id for each document
        doc_topic_ids = get_doc_topic(self.best_model, self.corpus)

        # create new topic_id feature in self.df_electronics
        self.df_electronics['topic_id'] = doc_topic_ids

        # assign topic names based on mapping dictionary
        self.df_electronics['new_topic_name'] = self.df_electronics['topic_id'].apply(lambda topic_id: topic_name_dict.get(topic_id, 'Unknown'))

        # create word cloud for each topic
        for i in range(self.best_model.num_topics):
            generate_word_cloud(self.best_model, i, topic_name_dict)

        # print documents with topic names and ids
        cols = ["reviews.text", "new_topic_name", "topic_id"]
        print(self.df_electronics[cols].head(15))

        # save pyLDAvis visualization to HTML file
        pyLDAvis.save_html(vis_data, 'lda_visualization.html')
        print('pyLDAvis representation saved as html file to working directory')


if __name__ == "__main__":
    lda_topic_model = LDATopicModel()
    lda_topic_model.preprocess_data()
    lda_topic_model.create_lemma_tokens()
    lda_topic_model.create_lemma_dictionary()
    lda_topic_model.generate_lda_models()
    lda_topic_model.compare_models()
    lda_topic_model.visualize_best_model()
