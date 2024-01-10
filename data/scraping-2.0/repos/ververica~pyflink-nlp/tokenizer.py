import spacy
from gensim.models.phrases import Phrases, Phraser
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess as spp
from gensim.parsing.preprocessing import STOPWORDS

def tokenize(msg):

    nlp = spacy.load('en', disable=['parser', 'ner'])

    msg = " ".join([token.lemma_ for token in nlp(msg)])

    tkn_list = (spp(str(msg), deacc=True))

    return tkn_list


def clean(tl):

    tl_stopwords = list(filter(lambda x: x not in STOPWORDS,tl))

    return tl_stopwords

def find_topic(m,model,dic):

    token_list = list(tokenize(m))

    texts = clean(token_list)

    vc = dic.doc2bow(texts)

    vector = model[vc]

    topics = sorted(vector, key=lambda x: x[1], reverse=True)

    return str(topics[0][0])
