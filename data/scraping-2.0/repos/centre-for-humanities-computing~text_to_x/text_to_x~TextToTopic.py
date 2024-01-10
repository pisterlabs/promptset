"""
"""
from gensim import models, corpora
from gensim.models.coherencemodel import CoherenceModel

# import pyLDAvis
# import pyLDAvis.gensim
# import matplotlib.pyplot as plt
# %matplotlib inline

from nltk.corpus import stopwords

from text_to_x.TextToX import TextToX
from text_to_x.TextToTokens import TextToTokens

nltk_lang_dict = {
    "hu": "hungarian",
    "sv": "swedish",
    "kk": "kazakh",
    "no": "norwegian",
    "fi": "finnish",
    "ar": "arabic",
    "in": "indonesian",
    "pt": "portuguese",
    "tr": "turkish",
    "az": "azerbaijani",
    "sl": "slovene",
    "es": "spanish",
    "da": "danish",
    "ne": "nepali",
    "ro": "romanian",
    "en": "english",
    "nl": "dutch",
    "tg": "tajik",
    "de": "german",
    "ru": "russian",
    "fr": "french",
    "it": "italian",
}


class TextToTokensWrapper(TextToTokens):
    def __init__(self, dfs, lang):
        """
        a wrapper solely for the covid-19 project
        """
        self.__dfs = dfs
        self.lang = lang

    def get_token_dfs(self):
        return self.__dfs


class TextToTopic(TextToX):
    def __init__(
        self,
        docs,
        tokentype="lemma",
        stopword_removal="nltk",
        pos_tags=["NOUN", "ADJ", "VERB", "PROPN" "ADV"],
        lowercase=True,
        lang=None,
        detect_lang_fun="polyglot",
        **kwargs,
    ):
        """
        docs (list, TextToTokens) list of string or object of type text to
        tokens.
        tokentype ('lemma'|'stem'|'token'): What token type should be used
        by default they are simplified to lemma, but can also be stem or
        token
        """
        super().__init__(lang=lang, kwargs=kwargs, detect_lang_fun=detect_lang_fun)

        if isinstance(docs, list) and isinstance(docs[0], str):
            self.ttt = TextToTokens(lang=self.lang, **kwargs)
        elif isinstance(docs, TextToTokens):
            self.ttt = docs
        else:
            ValueError(
                f"docs should be a list of string or a object of \
                         type TextToTokens, not a type {type(docs)}"
            )

        if tokentype is None:
            tokentype = "token"
        self.tokentype = tokentype
        self.stopword_removal = stopword_removal
        self.pos_tags = set(pos_tags)
        self.lang = self.ttt.lang
        self.tokenlists = [
            self.__extract_tok(df, self.tokentype, self.pos_tags)
            for df in self.ttt.get_token_dfs()
        ]
        if lowercase:
            self.tokenlists = [[t.lower() for t in tl] for tl in self.tokenlists]
        if stopword_removal == "nltk":
            self.__remove_stopwords_ntlk()
        else:
            raise ValueError(
                f"The method {stopword_removal} for removing stopwords is \
                               not currently implemented"
            )

    @staticmethod
    def __extract_tok(df, tokentype, pos_tags):
        if pos_tags:
            criteria = df["upos"].isin(pos_tags)
            return df[tokentype][criteria]
        else:
            return df[tokentype]

    def __remove_stopwords_ntlk(self):
        if isinstance(self.lang, str):
            if self.lang in nltk_lang_dict:
                lang = nltk_lang_dict[self.lang]
            else:
                ValueError(
                    f"The language code {self.lang} is not valid in \
                             NLTK"
                )
            stop_words = set(stopwords.words(lang))
            self.tokenlists = [
                [t for t in tl if t not in stop_words] for tl in self.tokenlists
            ]
        else:
            res = []
            for tl, la in zip(self.tokenlists, self.lang):
                if not res or la != lang:
                    lang = la
                    nltk_lang = nltk_lang_dict[lang]
                    stop_words = set(stopwords.words(nltk_lang))
                res.append([t for t in tl if t not in stop_words])
            self.tokenlists = res

    def train_topic(
        self,
        num_topics,
        no_below=1,
        no_above=0.9,
        keep_n=None,
        keep_tokens=None,
        remove_most_freq_n=None,
        bad_tokens=None,
        model="ldamulticore",
        bigrams=True,
        **kwargs,
    ):
        """
        no_below (int|None) – Keep tokens which are contained in at least
        no_below documents.
        no_above (float|None): Keep tokens which are contained in no
        more than no_above documents (fraction of total corpus size,
        not an absolute number).
        keep_n (int|None) – Keep only the first keep_n most frequent
        tokens.
        keep_tokens (iterable of str) – Iterable of tokens that must stay in
        dictionary after filtering.
        remove_most_freq_n (int|None): Remove n most frequent tokens
        model ('ldamulticore'|'lda'|'ldamallet')
        """
        if bigrams is True:
            phrases = models.Phrases(self.tokenlists, delimiter=b" ")
            phraser = models.phrases.Phraser(phrases)
            self.tokenlists = [phraser[tl] for tl in self.tokenlists]

        dictionary = corpora.Dictionary(self.tokenlists)

        if remove_most_freq_n:
            dictionary.filter_n_most_frequent(remove_most_freq_n)
        dictionary.filter_extremes(
            no_below=no_below, no_above=no_above, keep_n=keep_n, keep_tokens=keep_tokens
        )

        bows = [dictionary.doc2bow(tl) for tl in self.tokenlists]

        if bad_tokens:
            dictionary.filter_tokens(
                bad_ids=[dictionary.id2token[tok] for tok in bad_tokens]
            )

        self.bows = bows
        self.dictionary = dictionary
        if model == "ldamulticore":
            self.model = models.LdaMulticore(
                bows, num_topics=num_topics, id2word=dictionary, **kwargs
            )
        if model == "lda":
            self.model = models.LdaModel(
                bows, num_topics=num_topics, id2word=dictionary, **kwargs
            )

        if model == "ldamallet":
            raise ValueError("mallet is not yet implemented")
            # self.model = gensim.models.LdaMallet(path_to_mallet,
            #                                 t_tfidf,
            #                                 num_topics=num_topics,
            #                                 id2word=dictionary)

    def get_coherence(self, **kwargs):
        coherence_model_lda = CoherenceModel(
            model=self.model, texts=self.tokenlists, corpus=self.bows, **kwargs
        )
        return coherence_model_lda.get_coherence()

    def get_log_complexity(self):
        return self.model.log_perplexity(self.bows)

    # def vis_lda(self, notebook=True):
    #     if notebook:
    #         pyLDAvis.enable_notebook()
    #     vis = pyLDAvis.prepare(self.lda, self.bows, self.dictionary)
    #     return vis

    def get_lda(self):
        return self.lda

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = models.LdaModel.load(path)


# visualize
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)

if __name__ == "__main__":
    from text_to_x.utils import get_test_data

    texts = get_test_data(short_splits=False)
    texts = [texts[i : i + 500] for i in range(0, len(texts), 500)]
    ttt = TextToTokens()
    ttt.texts_to_tokens(texts)
    dfs = ttt.get_token_dfs()

    # test wrapper
    tttw = TextToTokensWrapper(dfs=dfs, lang=ttt.lang)

    topic = TextToTopic(tttw)
    topic.train_topic(10, no_below=1, iterations=100, passes=10)
    topic.get_coherence()  # maximize
    topic.get_log_complexity()
