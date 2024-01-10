import json
import pandas as pd
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models
import gensim.corpora as corpora

from gensim.models import CoherenceModel, ldamodel, TfidfModel


class TopicModeler:
    def __init__(self, file):
        self.file = file
        self.texts = None
        self.dictionary = None
        self.corpus = None
        self.tfidf = None
        self.model = None

    def set_texts(self):
        with open(self.file, 'r') as f:
            self.texts = json.load(f)

    def set_dictionary(self):
        self.dictionary = corpora.Dictionary(self.texts)

    def set_corpus(self):
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]

    def set_tfidf(self):
        self.tfidf = TfidfModel(self.corpus, smartirs='ntc')

    def print_readable_corpus(self):
        readable_corpus = [[(self.dictionary[id], freq) for id, freq in cp] for cp in self.corpus[:1]]
        print('\nPrinting readable corpus:')
        print(readable_corpus)

    def fit_lda(self, num_topics):
        self.model = ldamodel.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            random_state=100,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True)

        self.model

    def print_top_keywords_in_topic(self):
        # Print the Keyword in the each topics
        print('\nPrinting top keywords per topic:')
        print(self.model.print_topics())

    def print_model_coherence(self):
        coherence_model_lda = CoherenceModel(
            model=self.model,
            texts=self.texts,
            dictionary=self.dictionary,
            coherence='c_v')

        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)

    def optimal_model_search(self, start=2, stop=5, step=3, iterations=50):
        """
        Attribute to: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#4whatdoesldado
        In:
            dictionary : Gensim dictionary
            corpus : Gensim corpus
            texts : List of input texts
            stop : Max num of topics
        Out:
            model_list : List of LDA topic models
            coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, stop, step):
            model = ldamodel.LdaModel(corpus=self.tfidf, num_topics=num_topics, id2word=self.dictionary, iterations=iterations)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=self.texts, dictionary=self.dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    def plot_model_search_results(self, coherence_values, start, stop, step):
        x = range(start, stop, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()

    def set_optimal_model(self, model_list, optimal_index=0):
        self.model = model_list[optimal_index]

    def produce_doc_topic_summary_df(self):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(self.model[self.corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self.model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(self.texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

        doc_topic_summary = sent_topics_df.reset_index()
        doc_topic_summary.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

        return sent_topics_df, doc_topic_summary

    def produce_topic_summary_df(self, sent_topics_df):
        # Number of Documents for Each Topic
        topic_counts = sent_topics_df['Dominant_Topic'].value_counts()

        # Percentage of Documents for Each Topic
        topic_contribution = round(topic_counts/topic_counts.sum(), 4)

        # Topic Number and Keywords
        topic_num_keywords = sent_topics_df\
            .groupby(['Dominant_Topic', 'Topic_Keywords'])\
            .agg('count')\
            .reset_index()\
            .sort_values('Perc_Contribution', ascending=False)

        topic_num_keywords = topic_num_keywords[['Dominant_Topic', 'Topic_Keywords']]

        # Concatenate Column wise
        df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

        # Change Column names
        df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

        df_dominant_topics.dropna(inplace=True)

        topic_summary = df_dominant_topics\
            .groupby(['Dominant_Topic', 'Topic_Keywords'])\
            .agg('max')\
            .reset_index()

        return topic_summary

    def produce_doc_topic_matrix(self):
        '''
        out: pandas df of (doc, topic_num) where the cell is probability that doc_i is assoicated with topic_j
        '''
        lda_output = self.model

        # # Return topic distribution for the given document bow, as a list of (topic_id, topic_probability) 2-tuples.
        doc_topic_matrix = lda_output.get_document_topics(self.corpus, minimum_probability=None)
        doc_topic_matrix = pd.DataFrame(doc_topic_matrix)

        def extract_prob(x):
            if x is None:
                return 0.0
            else:
                return x[1]

        doc_topic_matrix = doc_topic_matrix.applymap(extract_prob)

        return doc_topic_matrix


# if __name__ == '__main__':
#     mod = TopicModeler('output.json')
#     mod.set_texts()
#     mod.set_dictionary()
#     mod.set_corpus()
#     mod.print_readable_corpus()
#     mod.fit_lda(num_topics=3)
#     mod.print_top_keywords_in_topic()
#     mod.print_model_coherence()
