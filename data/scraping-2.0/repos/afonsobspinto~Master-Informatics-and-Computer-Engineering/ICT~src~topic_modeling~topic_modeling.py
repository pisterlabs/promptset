import os
import pickle
import re
from datetime import datetime
from pprint import pprint
import gensim
import gensim.corpora as corpora
import pandas as pd
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
import spacy as spacy
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from settings import MALLET_PATH, MODELS_PATH
from utils import ENGLISH_STOPWORDS, log
import matplotlib.colors as mcolors
from wordcloud import WordCloud


def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)  # deacc=True removes punctuations


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in ENGLISH_STOPWORDS] for doc in texts]


class TopicModeling:
    def __init__(self, df, original_path):
        self.df = df
        self.original_path = original_path
        self.data = df.drop_duplicates().tweet.values.tolist()
        self.data_words = list(sent_to_words(self.data))
        self._generate_models()
        self._save_path()
        self.lda = None
        self.mod = None
        self.df_topic_keywords = None

    def _save_path(self):
        self.id = re.sub(r'-| |:|\.', '_', str(datetime.now()))
        self.save_path = f"{MODELS_PATH}/{self.id}"
        os.makedirs(self.save_path)

    def _generate_models(self):
        data_words_nostops = remove_stopwords(self.data_words)
        data_words_bigrams = self._make_bigrams(data_words_nostops)
        self.data_lemmatized = self._lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        self.id2word = corpora.Dictionary(self.data_lemmatized)
        texts = self.data_lemmatized
        self.corpus = [self.id2word.doc2bow(text) for text in texts]

    def model(self, method="mallet", num_topics=6, save=False):
        log(f"Modeling with {num_topics} num_topics")
        if method == "mallet":
            self.mod = self._lda_mallet(num_topics)
        else:
            self.mod = self._lda_model(num_topics)
        if save:
            self.save_lda()

    def _lda_mallet(self, num_topics):
        # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
        self.lda = gensim.models.wrappers.LdaMallet(MALLET_PATH, corpus=self.corpus,
                                                    num_topics=num_topics, id2word=self.id2word)
        return gensim.models.wrappers.ldamallet.malletmodel2ldamodel(self.lda)

    def _lda_model(self, num_topics):
        self.lda = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                                   id2word=self.id2word,
                                                   num_topics=num_topics,
                                                   random_state=100,
                                                   update_every=1,
                                                   chunksize=100,
                                                   passes=10,
                                                   alpha='auto',
                                                   per_word_topics=True)
        return self.lda

    def get_coherence(self):
        # a measure of how good the model is. lower the better.
        coherence_model_lda = CoherenceModel(model=self.lda, texts=self.data_lemmatized,
                                             dictionary=self.id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        return coherence_lda

    def _make_bigrams(self, texts):
        bigram = gensim.models.Phrases(self.data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        return [bigram_mod[doc] for doc in texts]

    @staticmethod
    def _lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        nlp = spacy.load('en', disable=['parser', 'ner'])
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def visualize(self, num_topics):
        if self.mod and self.lda:
            pprint(self.lda.print_topics())
            ldavis_data_filepath = os.path.join(self.save_path + '/ldavis_prepared_' + str(num_topics)
                                                + "_" + self.id)
            ldavis_prepared = pyLDAvis.gensim.prepare(self.mod, self.corpus, self.id2word)
            with open(ldavis_data_filepath, 'wb') as f:
                log("Dumping pyLDAvis")
                pickle.dump(ldavis_prepared, f)
            log("Saving pyLDAvis html")
            pyLDAvis.save_html(ldavis_prepared, ldavis_data_filepath + '.html')

    def compute_best_model(self, stop, start=2, step=3, show=True):
        log("Computing best model")
        coherence_values = []
        model_list = []
        for num_topics in range(start, stop, step):
            self.model(num_topics=num_topics)
            model_list.append(self.lda)
            coherence_values.append(self.get_coherence())
        best_index = coherence_values.index(max(coherence_values))
        num_topics = range(start, stop, step)[best_index]
        self.lda = model_list[best_index]
        if show:
            self.save_plot_coherence_scores(stop, start, step, coherence_values)
            self.print_coherence_values(stop, start, step, coherence_values)
            self.visualize(num_topics)
        self.save_lda()
        return num_topics

    def save_lda(self):
        log("Saving lda")
        self.lda.save(f"{self.save_path}/lda.model")

    def save_plot_coherence_scores(self, stop, start, step, coherence_values):
        x = range(start, stop, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend("coherence_values", loc='best')
        plt.savefig(f"{self.save_path}/{start}_{stop}_{step}.png")

    @staticmethod
    def print_coherence_values(stop, start, step, coherence_values):
        x = range(start, stop, step)
        for m, cv in zip(x, coherence_values):
            print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

    def format_topics_sentences(self):
        topics_df = pd.DataFrame()
        # Get main topic in each document
        for i, row in enumerate(self.lda[self.corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self.lda.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    topics_df = topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]),
                                                 ignore_index=True)
                else:
                    break

        # Add original text to the end of the output
        contents_ids = self._get_ids()
        contents = pd.Series(self.data)
        topics_df = pd.concat([topics_df, contents], axis=1)
        topics_df = pd.concat([topics_df, contents_ids], axis=1)
        topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic', 'Text', 'id']
        return topics_df

    def _get_ids(self):
        cols = ['id', 'tweet', 'user', 'date']
        original_data = pd.read_csv(self.original_path, names=cols)
        data = pd.merge(original_data, self.df, on="id").drop_duplicates().id.values.tolist()
        return pd.Series(data)

    def save_dominant_topics_per_sentence(self):
        log("Dominant topics per sentence")
        df_topic_keywords = self.get_topic_keywords_table()
        df_dominant_topic = df_topic_keywords.reset_index()
        df_dominant_topic.to_csv(f"{self.save_path}/dominant_topics_per_sentence.csv", index=False)
        log("Dominant topics per sentence saved")

    def save_representative_sentence_per_topic(self):
        log("Representative sentence per topic")
        df_topic_keywords = self.get_topic_keywords_table()
        topics_sorteddf_mallet = pd.DataFrame()
        stopics_outdf_grpd = df_topic_keywords.groupby('Dominant_Topic')
        for i, grp in stopics_outdf_grpd:
            topics_sorteddf_mallet = pd.concat([topics_sorteddf_mallet,
                                                grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], axis=0)
        topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
        topics_sorteddf_mallet.to_csv(f"{self.save_path}/representative_sentence_per_topic.csv", index=False)
        log("Representative sentence per topic saved")

    def get_topic_keywords_table(self):
        if self.df_topic_keywords is None:
            self.df_topic_keywords = self.format_topics_sentences()
        return self.df_topic_keywords

    def save_word_cloud(self, num_topics):
        pages = int(num_topics / 6)
        topics = self.mod.show_topics(formatted=False, num_topics=num_topics)

        index = 0
        for i in range(0, pages):
            cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
            cloud = WordCloud(stopwords=ENGLISH_STOPWORDS,
                              background_color='white',
                              width=2500,
                              height=1800,
                              max_words=10,
                              colormap='tab10',
                              color_func=lambda *args, **kwargs: cols[i],
                              prefer_horizontal=1.0)
            fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)
            for j, ax in enumerate(axes.flatten()):
                fig.add_subplot(ax)
                topic_words = dict(topics[index][1])
                to_del = []
                for key, value in topic_words.items():
                    if value == 0.0:
                        to_del.append(key)
                for k in to_del:
                    del topic_words[k]
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                plt.gca().set_title('Topic ' + str(index), fontdict=dict(size=16))
                plt.gca().axis('off')
                index += 1

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.axis('off')
            plt.margins(x=0, y=0)
            plt.tight_layout()
            plt.savefig(f"{self.save_path}/wordcloud{i}.png")
