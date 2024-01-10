import sys
import json
import os
import time
from vaderSentiment import vaderSentiment
import xiaohan_sentiment
from senti_classifier import senti_classifier
from openai_encoder import Model
from stanfordcorenlp import StanfordCoreNLP
from textblob.en.sentiments import NaiveBayesAnalyzer, PatternAnalyzer
from tqdm import tqdm

# Usage: python sentiment_barebones.py filename no.articles (Sentiment scorer comparison)
# The output files of this are used for sentiment_eval.eval_sentiment's input files

openai_time = time.time()
openai_model = Model()
print('\nLoading OpenAI sentiment model took', time.time() - openai_time, 'seconds\n')

vader_analyser = vaderSentiment.SentimentIntensityAnalyzer()
nba = NaiveBayesAnalyzer()
pa = PatternAnalyzer()


def sentiment(fname, max_articles=10):
    """
    Format: python sentiment.py filename
    """
    start_time = time.time()

    if not os.path.exists('./out-sentiment/'):
        os.makedirs('./out-sentiment/')
    in_path = './out-parse/' + fname
    out_path = './out-sentiment/' + fname

    # You need to download this from https://stanfordnlp.github.io/CoreNLP/. Too big to upload to GitHub.
    stanford_nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-02-27', memory='8g')

    f_in = open(in_path, 'r')
    count, total = 0, sum(1 for line in f_in)
    f_in.close()

    f_in = open(in_path, 'r')
    f_out = open(out_path, 'w')
    f_out_time = open('./out-sentiment/times.csv', 'a')

    for line in f_in:
        article_time = time.time()
        scorer_times = []

        to_write = json.loads(line)
        if to_write['num_relevant_sentences'] == 0:
            count += 1
            print(count, "/", total, "articles analysed (last article skipped)")
            continue

        to_write['relevance_score_sents'] = to_write['num_relevant_sentences'] / to_write['num_sentences']
        to_write['relevance_score_keyword_rank'] = 1 / to_write['keyword_rank']  # Assume Zipf distribution with s ~= 1

        sents = to_write['relevant_sentences']

        # VADER: rule-based / lexical (https://github.com/cjhutto/vaderSentiment)
        score_time = time.time()
        for sent in sents:
            sents[sent]['sentiment_score_vader'] = vader_analyser.polarity_scores(sent)['compound']  # y = -1 to 1
        # print(time.time() - score_time, 'seconds to analyse article using VADER')
        scorer_times.append(time.time() - score_time)

        # XiaoHan: Convolutional Neural Network, trained on Twitter (github.com/xiaohan2012/twitter-sent-dnn)
        score_time = time.time()
        for sent in sents:
            sents[sent]['sentiment_score_xiaohan'] = (xiaohan_sentiment.sentiment_score(sent) - 0.5) * 2  # y = 0 to 1
        # print(time.time() - score_time, 'seconds to analyse article using XiaoHan')
        scorer_times.append(time.time() - score_time)

        # K. Cobain: MaxEnt / Naive Bayes, trained on SentiWordNet (github.com/kevincobain2000/sentiment_classifier)
        # Note: slow (~2 seconds per sentence)
        score_time = time.time()
        for sent in sents:
            pos_score, neg_score = senti_classifier.polarity_scores([sent])
            sents[sent]['sentiment_score_kcobain'] = pos_score - neg_score  # y = normal distribution mean=0, unbounded
        # print(time.time() - score_time, 'seconds to analyse article using Kevin Cobain')
        scorer_times.append(time.time() - score_time)

        # OpenAI: mLSTM-based, trained on Amazon reviews (github.com/openai/generating-reviews-discovering-sentiment)
        # Note: VERY slow (~7 seconds per sentence, faster with larger batches). May be faster with tensorflow-gpu.
        batched_sents = []  # OpenAI runs faster with batched sentences
        for sent in sents:
            batched_sents.append(sent)  # OpenAI runs faster with batched sentences
        score_time = time.time()
        try:
            openai_sentiment = openai_model.transform(batched_sents)[:, 2388]
            for i in range(len(batched_sents)):
                sents[batched_sents[i]]['sentiment_score_openai'] = float(openai_sentiment[i])
                # y = normal distribution with mean=0, unbounded
        except ValueError:  # tensorflow bugs
            for i in range(len(batched_sents)):
                sents[batched_sents[i]]['sentiment_score_openai'] = 'ERROR'
        scorer_times.append(time.time() - score_time)

        # StanfordNLP (https://nlp.stanford.edu/sentiment/). Classifier instead of regressor (y = [1,2,3,4,5])
        # Can be batched, but may not split sentences consistently at the same points as SpaCy (and runs quick anw)
        score_time = time.time()
        for sent in sents:
            try:
                props = {'annotators': 'sentiment', 'outputFormat': 'json', 'timeout': 15000}
                res = json.loads(stanford_nlp.annotate(sent, properties=props))
                sents[sent]['sentiment_score_stanford'] = (float(res['sentences'][0]['sentimentValue']) - 2.0) / 2.0
            except json.decoder.JSONDecodeError:
                sents[sent]['sentiment_score_stanford'] = 'ERROR'

        # print(time.time() - score_time, 'seconds to analyse article using StanfordNLP')
        scorer_times.append(time.time() - score_time)

        # TextBlob. Uses Naive Bayes analyzer trained on the movie review corpus (textblob.readthedocs.io/en/dev/)
        # Can be batched, but may not split sentences consistently at the same points as SpaCy (and runs quick anw)
        score_time = time.time()
        for sent in sents:
            res = nba.analyze(sent)
            sents[sent]['sentiment_score_textblob_bayes'] = res[1] - res[2]
        # print(time.time() - score_time, 'seconds to analyse article using TextBlob (Naive Bayes)')
        scorer_times.append(time.time() - score_time)

        # TextBlob. Uses pattern analyser (https://textblob.readthedocs.io/en/dev/)
        # Can be batched, but may not split sentences consistently at the same points as SpaCy (and runs quick anw)
        score_time = time.time()
        for sent in sents:
            sents[sent]['sentiment_score_textblob'] = pa.analyze(sent).polarity
        # print(time.time() - score_time, 'seconds to analyse article using TextBlob (Pattern Analyser)')
        scorer_times.append(time.time() - score_time)

        # 'summarise' sentiment score of an article via weighted average of each sentence
        # TODO measure relevance score of each sentence & use it as weight for sentiment score? instead of keyword_count
        sentiment_score_labels = ['vader', 'xiaohan', 'kcobain', 'openai', 'stanford', 'textblob', 'textblob_bayes']
        for label in sentiment_score_labels:
            full_label = 'sentiment_score_' + label
            weighted_avg, keyword_total_count = 0.0, 0
            for sent in sents:
                if sents[sent][full_label] != 'ERROR':
                    weighted_avg += sents[sent][full_label] * sents[sent]['keyword_count']
                    keyword_total_count += sents[sent]['keyword_count']
            try:
                to_write[full_label] = weighted_avg / keyword_total_count
            except ZeroDivisionError:
                to_write[full_label] = 'ERROR'

        f_out_time.write('\n')
        f_out_time.write(",".join([str(t) for t in scorer_times]))

        f_out.write(json.dumps(to_write))
        f_out.write('\n')

        count += 1
        if count >= max_articles:
            break
        print(count, "/", total, "articles analysed (last article =", time.time() - article_time, "seconds)")

    stanford_nlp.close()
    f_in.close()
    f_out.close()
    f_out_time.close()

    full_time = time.time() - start_time
    print('Sentiment analyses took', int(full_time // 60), 'minutes', full_time % 60, 'seconds')


def sentiment_openai(fname):
    start_time = time.time()

    if not os.path.exists('./out-sentiment-openai/'):
        os.makedirs('./out-sentiment-openai/')
    in_path = './out-parse/' + fname
    out_path = './out-sentiment-openai/' + fname

    f_in = open(in_path, 'r')
    i, num_articles = 0, 0
    batched_sentences = []
    sentence_index = {}
    for line in f_in:
        data = json.loads(line)
        sents = data['relevant_sentences']
        for sent in sents:
            batched_sentences.append(sent)
            sentence_index[sent] = i
            i += 1
        num_articles += 1
    f_in.close()

    # OpenAI: mLSTM-based, trained on Amazon reviews (github.com/openai/generating-reviews-discovering-sentiment)
    # Note: VERY slow (~7 seconds per sentence, faster with larger batches). May be faster with tensorflow-gpu.
    sentiment_vector = openai_model.transform(batched_sentences)[:, 2388]

    f_in = open(in_path, 'r')
    f_out = open(out_path, 'w')
    for line in f_in:
        to_write = json.loads(line)
        if to_write['num_relevant_sentences'] == 0:
            continue

        to_write['relevance_score_sents'] = to_write['num_relevant_sentences'] / to_write['num_sentences']
        to_write['relevance_score_keyword_rank'] = 1 / to_write['keyword_rank']  # Assume Zipf distribution with s ~= 1

        sents = to_write['relevant_sentences']
        for sent in sents:
            sents[sent]['sentiment_score'] = float(sentiment_vector[sentence_index[sent]])

        # 'summarise' sentiment score of an article via weighted average of each sentence
        weighted_avg, keyword_total_count = 0.0, 0
        for sent in sents:
            if sents[sent]['sentiment_score'] != 'ERROR':
                weighted_avg += sents[sent]['sentiment_score'] * sents[sent]['keyword_count']
                keyword_total_count += sents[sent]['keyword_count']
        try:
            to_write['sentiment_score'] = weighted_avg / keyword_total_count
        except ZeroDivisionError:
            to_write['sentiment_score'] = 'ERROR'

        f_out.write(json.dumps(to_write))
        f_out.write('\n')

    f_in.close()
    f_out.close()

    full_time = time.time() - start_time
    # print('Sentiment analyses took', int(full_time // 60), 'minutes', full_time % 60, 'seconds')
    # print('Per article:', full_time / num_articles, 'seconds')
    # print('Per sentence:', full_time / len(batched_sentences), 'seconds')


def sentiment_vader(fname):
    if not os.path.exists('./out-sentiment-vader/'):
        os.makedirs('./out-sentiment-vader/')
    in_path = './out-parse/' + fname
    out_path = './out-sentiment-vader/' + fname

    f_in = open(in_path, 'r')
    f_out = open(out_path, 'w')

    for line in tqdm(f_in):
        to_write = json.loads(line)
        if to_write['num_relevant_sentences'] == 0:
            continue

        to_write['relevance_score_sents'] = to_write['num_relevant_sentences'] / to_write['num_sentences']
        to_write['relevance_score_keyword_rank'] = 1 / to_write['keyword_rank']  # Assume Zipf distribution with s ~= 1

        sents = to_write['relevant_sentences']

        # VADER: rule-based / lexical (https://github.com/cjhutto/vaderSentiment)
        for sent in sents:
            sents[sent]['sentiment_score'] = vader_analyser.polarity_scores(sent)['compound']  # y = -1 to 1

        # 'summarise' sentiment score of an article via weighted average of each sentence
        weighted_avg, keyword_total_count = 0.0, 0
        for sent in sents:
            if sents[sent]['sentiment_score'] != 'ERROR':
                weighted_avg += sents[sent]['sentiment_score'] * sents[sent]['keyword_count']
                keyword_total_count += sents[sent]['keyword_count']
        try:
            to_write['sentiment_score'] = weighted_avg / keyword_total_count
        except ZeroDivisionError:
            to_write['sentiment_score'] = 'ERROR'

        f_out.write(json.dumps(to_write))
        f_out.write('\n')

    f_in.close()
    f_out.close()


if __name__ == '__main__':
    sentiment(sys.argv[1], int(sys.argv[2]))
