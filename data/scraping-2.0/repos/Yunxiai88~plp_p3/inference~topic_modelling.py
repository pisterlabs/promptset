import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Gensim
import gensim, spacy
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from nltk.corpus import stopwords
# if you dont have stop words yet
# nltk.download('stopwords')

from collections import Counter


class TopicModeling:

    def __init__(self, df: pd.DataFrame, sentiment_type: str,sentiment_column_name:str, review_column_name: str):
        self.df_ur = df[df[sentiment_column_name] == sentiment_type].copy()
        self.df_ur.dropna(subset=[sentiment_column_name,review_column_name],inplace=True)
        self.review_column_name = review_column_name

    def pre_processing(self):
        print("pre_processing in progress")
        review_column = self.review_column_name
        sentences = self.df_ur[review_column].astype(str).values.tolist()
        data_words = []
        for sent in sentences:
            sent = re.sub('\s+', ' ', sent)  # remove newline chars
            sent = re.sub("\'", "", sent)  # remove single quotes
            sent = gensim.utils.simple_preprocess(str(sent), deacc=True)  # deacc=True removes punctuation
            data_words.append(sent)

        stop_words = stopwords.words('english')
        stop_words.extend(['game', 'good', 'love', 'great', 'app'])

        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=2, threshold=100)  # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

        # Faster way to get a sentence clubbed as a trigram/bigram

        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        # Remove stopWords
        data_words_nostops = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in
                              data_words]

        data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        data_lemmatized = []
        for sent in data_words_bigrams:
            doc = nlp(" ".join(sent))
            data_lemmatized.append([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']])
        self.data_lemmatized = data_lemmatized
        print("pre_processing is done!")
        return data_lemmatized

    def topic_model(self, number_of_topics, data_lemmatized=None):
        # Create Dictionary
        print("start to create topic model")
        if data_lemmatized is None:
            data_lemmatized = self.pre_processing()

        id2word = corpora.Dictionary(data_lemmatized)
        # Create Corpus
        texts = data_lemmatized
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=number_of_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=10,
                                                    passes=10,
                                                    alpha='symmetric',
                                                    iterations=100,
                                                    per_word_topics=True)
        # pprint(lda_model.print_topics())
        # doc_lda = lda_model[corpus]
        # # Compute Perplexity
        # print('\nPerplexity: ',
        #       lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
        # # Compute Coherence Score
        # coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word,
        #                                      coherence='c_v')
        # coherence_lda = coherence_model_lda.get_coherence()
        # print('\nCoherence Score: ', coherence_lda)
        print("finish create topic model")
        return lda_model, corpus


    def plot_word_of_importance_chart(self, lda_model, data_lemmatized, file_name):
        print("start ploting word of importance chart")
        topics = lda_model.show_topics(formatted=False)
        data_ready = data_lemmatized
        data_flat = [w for w_list in data_ready for w in w_list]
        counter = Counter(data_flat)
        out = []
        for i, topic in topics:
            for word, weight in topic:
                out.append([word, i, weight, counter[word]])
        
        df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])
        
        # Plot Word Count and Weights of Topic Keywords
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharey=True, dpi=160)
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        for i, ax in enumerate(axes.flatten()):
            ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
                   label='Word Count')
            ax_twin = ax.twinx()
            ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                        label='Weights')
            ax.set_ylabel('Word Count', color=cols[i])
            ax_twin.set_ylim(0, 0.030);
            ax.set_ylim(0, 30000)
            ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
            ax.tick_params(axis='y', left=False)
            ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right')
            ax.legend(loc='upper left');
            ax_twin.legend(loc='upper right')

        fig.tight_layout(w_pad=2)
        fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
        fig.savefig(file_name)
