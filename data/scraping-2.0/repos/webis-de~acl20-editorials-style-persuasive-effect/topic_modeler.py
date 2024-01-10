import pandas as pd
import gensim
import text_miner
from gensim import corpora, models
from gensim.models import CoherenceModel
import numpy as np

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
class topic_modeler():
    no_below = 50
    no_above = 0.40
    keep_n = 10000
    num_topics  = 4
    lda_model = None

    def __init__(self, data):
        self.data = data
        self.no_below = int(len(data) * 0.05)

        self.id2word = None
        self.corpus = None
        self.data_lemmatized = None
        self.lda_model = None
    


    def get_term_doc_frequency(self):
        self.preprocesser = text_miner.topic_modeling_preprocess(self.data)
        self.data_lemmatized = self.preprocesser.data_lemmatized
        # Create Dictionary
        self.id2word = corpora.Dictionary(self.data_lemmatized)

        # Term Document Frequency
        self.corpus = [self.id2word.doc2bow(text) for text in self.data_lemmatized]

    def show_human_readable_freq(self):
        # Human readable format of corpus (term-frequency)
        return [[(self.id2word[id], freq) for id, freq in cp] for cp in self.corpus[:1]]


    def compute_coherence_values(self, limit, start=2, step=3, lda_type="default", show_details=False):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics
        lda_type : default or mallet
        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        self.get_term_doc_frequency()
        coherence_values = []
        model_list = []
        topics_num_arr = []
        mallet_path = 'thirdparty-data/mallet-2.0.8/mallet-2.0.8/bin/mallet' # update this path

        for num_topics in range(start, limit, step):
            model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                            id2word=self.id2word,
                                            num_topics=num_topics, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=50,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True) if lda_type=="default" else gensim.models.wrappers.LdaMallet(mallet_path, corpus=self.corpus, num_topics=num_topics, id2word=self.id2word)
            model_list.append(model)
            topics_num_arr.append(num_topics)
            coherencemodel = CoherenceModel(model=model, texts=self.data_lemmatized, dictionary=self.id2word, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
        optimal_idx = np.argmax(coherence_values)
        self.lda_model = model_list[optimal_idx]
        print("optimal model has a coherence value of ", round(coherence_values[optimal_idx], 2), ' and # topics: ', (topics_num_arr[optimal_idx]))
        
        # Visualize
        topic_modeler.show_coherence_vals_graph(coherence_values, limit, start=start, step=step )

        if show_details:
            for m, cv in zip(x, coherence_values):
                print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
        return model_list, coherence_values

    ## VISIULISATION
    @staticmethod
    def show_coherence_vals_graph(coherence_values, limit, start=2, step=3):
        
        x = range(start, limit, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()

    def lda(self, base="tftidf"):
        processed_docs = [text_miner.preprocess(x, stem_lemma=3)['lemmas'] for x in self.data]
        self.id2word = corpora.Dictionary(processed_docs)


        '''
        Filter out tokens that appear in

            less than 15 documents (absolute number) or
            more than 0.5 documents (fraction of total corpus size, not absolute number).
            after the above two steps, keep only the first 100000 most frequent tokens.
        '''
        self.id2word.filter_extremes(no_below= self.no_below, no_above= self.no_above, keep_n=self.keep_n)

        '''
        Gensim doc2bow
        For each document we create a dictionary reporting how many
        words and how many times those words appear. Save this to ‘bow_corpus’, then check our selected document earlier.
        '''
        self.corpus = [self.id2word.doc2bow(doc) for doc in processed_docs]
                
        if base == "tftidf":
            tfidf = models.TfidfModel(self.corpus)
            corpus_tfidf = tfidf[self.corpus]
            self.lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=self.num_topics, id2word=self.id2word, passes=50, workers=4)
        elif base == "bow":
            
            self.lda_model = gensim.models.LdaMulticore(self.corpus, num_topics=self.num_topics, id2word=self.id2word, passes=50, workers=4)
        return self.lda_model

    def print_topics(self):
        if self.lda_model is not None:
            print(self.lda_model.print_topics(num_topics=self.num_topics, num_words=5))

    def visualize_topics_interactively(self):
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(self.lda_model, self.corpus, self.id2word)
        return vis


    def get_doc_dominant_topic(self):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(self.lda_model[self.corpus]):
            row =row[0]
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self.lda_model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(self.data)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return(sent_topics_df)

    def get_most_represented_doc_per_topic(self):
        # Group top 5 sentences under each topic
        result_df = pd.DataFrame()

        doc_topics_outdf_grpd = self.get_doc_dominant_topic().groupby('Dominant_Topic')

        for i, grp in doc_topics_outdf_grpd:
            result_df = pd.concat([result_df, grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                                    axis=0)

        # Reset Index    
        result_df.reset_index(drop=True, inplace=True)

        # Format
        result_df.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
        return result_df

    def get_topic_distr(self):
        topic_docs_keywords_df = self.get_doc_dominant_topic()
        # Number of Documents for Each Topic
        topic_counts = topic_docs_keywords_df['Dominant_Topic'].value_counts()

        # Percentage of Documents for Each Topic
        topic_contribution = round(topic_counts/topic_counts.sum(), 4)

        # Topic Number and Keywords
        topic_num_keywords = topic_docs_keywords_df[['Dominant_Topic', 'Topic_Keywords']]

        # Concatenate Column wise
        result_df = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

        # Change Column names
        result_df.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

        return result_df

    def show_word_cloud(self, limit=10):
        # 1. Wordcloud of Top N words in each topic
       

        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(stopwords=self.preprocesser.stop_words,
                        background_color='white',
                        width=2500,
                        height=1800,
                        max_words=10,
                        colormap='tab10',
                        color_func=lambda *args, **kwargs: cols[i],
                        prefer_horizontal=1.0)

        topics = self.lda_model.show_topics(formatted=False)

        fig, axes = plt.subplots( (limit/2), 2, figsize=(10,10), sharex=True, sharey=True)

        
        for i, ax in enumerate(axes.flatten()):
            fig.add_subplot(ax)
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
            plt.gca().axis('off')
            if i> limit:
                break


        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.show()

    


    def topics_per_document(self):
        dominant_topics = []
        topic_percentages = []
        for i, corp in enumerate(self.corpus):
            topic_percs, wordid_topics, wordid_phivalues = self.lda_model[corp]
            dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
            dominant_topics.append((i, dominant_topic))
            topic_percentages.append(topic_percs)

        # Total Topic Distribution by actual weight
        topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
        df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

        # Distribution of Dominant Topics in Each Document
        df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
        dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
        df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()
        return(df_dominant_topic_in_each_doc, df_topic_weightage_by_doc)


    def show_doc_topic_barcharts(self):
        from matplotlib.ticker import FuncFormatter

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120, sharey=True)

        # Topic Distribution by Dominant Topics
        df_dominant_topic_in_each_doc,  df_topic_weightage_by_doc= self.topics_per_document()

        # Top 3 Keywords for each Topic
        topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False) 
                                        for j, (topic, wt) in enumerate(topics) if j < 3]

        df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
        df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
        df_top3words.reset_index(level=0,inplace=True)


        ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
        ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
        tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
        ax1.xaxis.set_major_formatter(tick_formatter)
        ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
        ax1.set_ylabel('Number of Documents')
        ax1.set_ylim(0, 1000)

        # Topic Distribution by Topic Weights
        ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
        ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
        ax2.xaxis.set_major_formatter(tick_formatter)
        ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))

        plt.show()