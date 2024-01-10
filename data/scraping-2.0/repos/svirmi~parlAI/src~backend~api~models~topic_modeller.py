import re
import string

import gensim
import pandas as pd
import spacy
from gensim import corpora
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from nltk.stem.snowball import DutchStemmer
import pickle


class TopicModeller:
    num_topics = 10
    mallet_path: string = ''
    corpus = None
    dictionary: corpora.Dictionary = None
    model: gensim.models.ldamodel.LdaModel = None
    documents = None
    cleanedDocuments = None
    modelpath: string = None
    corpuspath: string = None
    dictpath: string = None

    def __init__(self, mallet_path, num_topics, corpuspath, dictpath, modelpath):
        self.num_topics = num_topics
        self.mallet_path = mallet_path
        self.corpuspath = corpuspath
        self.modelpath = modelpath
        self.dictpath = dictpath

        self.stopwords = stopwords.words('dutch')
        extra = ['mening', 'gevolgen', 'vragen', 'stelling', 'bericht', 'bekend', 'bereid', 'voornemens']
        self.stopwords.extend(extra)
        self.stopwords = set(self.stopwords)

        self.stemmer = DutchStemmer()

        # If the corpus and the model exist in the disk, load them.
        try:
            self.model = gensim.models.ldamodel.LdaModel.load(modelpath)
        except FileNotFoundError:
            pass

        try:
            self.dictionary = corpora.Dictionary.load(dictpath)
            # with open(dictpath, 'rb') as file:
            #     self.dictionary = pickle.load(file)
        except FileNotFoundError:
            pass

        try:
            with open(corpuspath, 'rb') as file:
                self.corpus = pickle.load(file)
        except FileNotFoundError:
            pass

    def clean_doc(self, doc):
        cleaned = " ".join(
            [i for i in doc.lower().split() if i not in self.stopwords and re.match(r'[a-zA-Z\-][a-zA-Z\-]{2,}', i)])
        cleaned = "".join(
            ch for ch in cleaned if ch not in set(string.punctuation))
        cleaned = re.sub(r'^https?:\/\/.*[\r\n]*', '', cleaned, flags=re.MULTILINE)
        cleaned = " ".join(token for token in self.lemmatize_doc(cleaned, ['NOUN', 'ADJ']) if token)
        cleaned = cleaned.replace('”', '')

        return cleaned

    def lemmatize_doc(self, doc, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        nlp = spacy.load('nl_core_news_sm', disable=['parser', 'ner'])

        text = nlp(doc)
        lemmatized_doc = [token.lemma_ for token in text if token.pos_ in allowed_postags]
        return lemmatized_doc

    def create_corpus(self):
        self.corpus = [self.dictionary.doc2bow(text) for text in self.cleanedDocuments]
        with open(self.corpuspath, 'wb') as f:
            pickle.dump(self.corpus, f)

    def create_dictionary(self, documents):
        self.dictionary = corpora.Dictionary(documents)
        self.dictionary.save(self.dictpath)

    def fit_model(self, documents):
        # Clean questions
        cleaned = [self.clean_doc(q).split() for q in documents]
        self.cleanedDocuments = cleaned

        # Build a Dictionary - association word to numeric id
        self.create_dictionary(cleaned)
        # Transform the collection of texts to a numerical form
        self.create_corpus()

        self.model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                                     id2word=self.dictionary,
                                                     num_topics=self.num_topics,
                                                     random_state=123,
                                                     chunksize=100,
                                                     passes=10,
                                                     alpha='auto',
                                                     per_word_topics=True)

        if self.modelpath:
            self.model.save(self.modelpath)

    def fit_model_mallet(self, documents):
        # Clean questions
        cleaned = [self.clean_doc(q).split() for q in documents]
        self.cleanedDocuments = cleaned

        # Build a Dictionary - association word to numeric id
        self.create_dictionary(cleaned)
        # Transform the collection of texts to a numerical form
        self.create_corpus()

        self.model = gensim.models.wrappers.LdaMallet(self.mallet_path,
                                                      corpus=self.corpus,
                                                      num_topics=self.num_topics,
                                                      id2word=self.dictionary,
                                                      alpha='auto')

        if self.modelpath:
            self.model.save(self.modelpath)

    def print_topics_simple(self):
        x = self.model.show_topics(num_topics=self.num_topics,
                                   num_words=10, formatted=False)
        topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]

        # Below Code Prints Topics and Words
        for topic, words in topics_words:
            print(str(topic) + ":" + str(words))

    def get_topics_scores(self):
        return self.model.print_topics()

    def get_coherence_score(self, documents):
        # Clean questions
        if not self.cleanedDocuments:
            self.cleanedDocuments = [self.clean_doc(q).split() for q in documents]

        coherence_model = CoherenceModel(
            model=self.model, texts=self.cleanedDocuments, dictionary=self.dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        print('\nCoherence Score: ', coherence)
        return coherence

    def get_topic_of_document(self, document):
        cleaned = self.clean_doc(document).split()
        bow = self.dictionary.doc2bow(cleaned)

        topic_dist = self.model.get_document_topics(bow)
        sorted_topics = sorted(topic_dist, key=lambda x: (x[1]), reverse=True)
        topic = {}
        topic_keywords = ''
        for j, (topic_num, prop_topic) in enumerate(sorted_topics):
            if j == 0:  # => dominant topic
                wp = self.model.show_topic(topic_num)
                topic['num'] = topic_num
                topic_keywords = ", ".join([word for word, prop in wp])

        topic['keywords'] = topic_keywords
        return topic
        # return topic_keywords

    def get_topics_per_document(self):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(self.model[self.corpus]):
            row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self.model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(self.documents)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return sent_topics_df
