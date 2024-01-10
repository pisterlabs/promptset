import sys
import os
sys.path.append("..")
from globals import ROOT_DIR
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import spacy
import numpy as np
from collections import Counter
from bert_embedding import BertEmbedding
import time


def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def get_scores(docs, nlp, seed):
    try:
        data_lemmatized = lemmatization(docs, nlp, allowed_postags=['NOUN'])

        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                                            id2word=id2word,
                                                            num_topics=20,
                                                            random_state=seed,
                                                            passes=2)
        # # Print the Keyword in the 10 topics
        dominant_keywords = []
        for i, row_list in enumerate(lda_model[corpus]):
            row = row_list[0] if lda_model.per_word_topics else row_list
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = lda_model.show_topic(topic_num)
                    dominant_keywords.extend([word for word, prop in wp])
                else:
                    break
        topic_words = Counter([keyword for keyword in dominant_keywords if keyword not in ['-PRON-']]).most_common(10)

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        return lda_model.log_perplexity(corpus), coherence_lda, [word for (word, count) in topic_words]
    except:
        return None, None, None


def bert_embed_words(words, bert_embedding):
    result = []
    items = bert_embedding(words)
    for item in items:
        try:
            result.append(item[1][0])
        except:
            result.append(np.zeros((768,)))
    return result


def aggregate(start, end, file_names):
    aggregate_data = {}
    for i in range(start, end):
        path_name = os.path.join(ROOT_DIR, 'data/{}_{}.npz'.format(file_names, i))
        results = np.load(path_name, allow_pickle=True)
        print("Downloading {}, Processed {} / {}".format(path_name, i, end - start))
        results = results['a']
        results = results[()]
        aggregate_data = {**results, **aggregate_data}
    return aggregate_data


def add_scores(embeds, word_count, scores):
    features = np.array([scores for _ in range(word_count)])  # adding 1
    embed = np.concatenate((embeds, features), -1)
    return embed


start = time.time()
timelines = aggregate(1, 18, 'timeline/user_timeline_processed')
print(len(timelines))
user_scores = {}
bert_topic_words = {}

nlp = spacy.load('en', disable=['parser', 'ner'])
bert_embedding = BertEmbedding()

print("Beginning LDA Analysis for both bert & lda")
save_count = 0
for i, (key, value) in enumerate(timelines.items()):
    if i % 100 == 0:
        print("{}) {} min".format(i, (time.time() - start)/60))

    value = value[:200]
    doc = value
    if len(doc) == 0:
        perplexity = 0
        coherence = 0
        topic_words = [' '] * 10
    else:
        perplexity, coherence, topic_words = get_scores(doc, nlp, 28)

    if perplexity is None:
        perplexity = 0
        coherence = 0
        topic_words = [' '] * 10

    user_scores[key] = [perplexity, coherence, topic_words]
    bert_embed = bert_embed_words(topic_words, bert_embedding)
    bert_embed_processed = add_scores(bert_embed, 10, [perplexity, coherence])
    bert_topic_words[key] = bert_embed_processed

    if i % 1000 == 0 and i != 0:
        np.savez(os.path.join(ROOT_DIR, 'data/lda/user_lda_scores_{}.npz'.format(save_count)), a=user_scores)
        np.savez(os.path.join(ROOT_DIR, 'data/bert/bert_topics_{}.npz'.format(save_count)), a=bert_topic_words)
        save_count += 1
        user_scores = {}
        bert_topic_words = {}

np.savez(os.path.join(ROOT_DIR, 'data/lda/user_lda_scores_{}.npz'.format(save_count)), a=user_scores)
np.savez(os.path.join(ROOT_DIR, 'data/bert/bert_topics_{}.npz'.format(save_count)), a=bert_topic_words)
