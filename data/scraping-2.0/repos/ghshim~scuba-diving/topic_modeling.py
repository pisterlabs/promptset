"""
Topic Modeling을 통해서 주어진 문서의 세부 토픽(주제)를 분류합니다. 세부 토픽의 개수는 설정할 수 있으며, default는 10입니다.
158-164번째 줄을 주석해제하면 perplexity와 coherence 결과에 따른 최적의 토픽 개수를 확인할 수 있습니다.

해당 코드는 pyLDAvis 라이브러리를 필요로 합니다.
설치 명령어: pip install pyLDAvis

Args:
    -d: data(document) path (required)
    -s: path to save t-SNE graph (required)
    -n: the number of topics (default 10)

Returns:
    Saved Path: ./figure/topic name/ 
    Name of files: '/topic' + save_path[-1] + '_TopicModeling.html' (html)

    The result of topic modeling using pyLDAvis

입력 예시:
    python topic_modeling.py -d './data/topic2/trust_robot.csv' -s '/topic2'
"""

import sys
import os
import itertools
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

sys.path.insert(0, '../')
import preprocess_csv as preprocess


def tokenizer(text):
    '''
    apply tokenizer
    '''
    cachedStopWords = stopwords.words("english")
    RegTok = RegexpTokenizer("[\w']{3,}")
    english_stops = set(stopwords.words('english'))
    tokens = RegTok.tokenize(text.lower())
    # stopwords 제외
    words = [word for word in tokens if (word not in english_stops) and len(word) > 2]
    
    stemmer = PorterStemmer()
    word_token = [stemmer.stem(i) for i in words]
    
    return words, word_token


def store_stem_result(texts, orig_words):
    filename = '/topic' + save_path[-1] + '_StemResult.csv'
    # make list 2 dim to 1 dim
    texts= list(itertools.chain(*texts))
    orig_words = list(itertools.chain(*orig_words))

    # store them in DataFrame
    df = pd.DataFrame()
    df['afterStem'] = texts
    df['beforeStem'] = orig_words
    # drop the same result of stem 
    df = df.drop_duplicates(['afterStem'])
    # sort by stem
    df = df.sort_values(by='afterStem', axis=0)

    df.to_csv('./data' + save_path + filename, index=False)



def save_targets(model, corpus):
    filename = '/topic' + save_path[-1] + '_predicted.csv'
    predicted = []
    for i in range(len(corpus)):
        y = model.get_document_topics(corpus)[i][0][0]
        predicted.append(y)

    df = pd.read_csv(data_path)
    df = df[['title', 'abstract']]
    df['predictedTopic'] = predicted
    df.to_csv('./data' + save_path + filename)


def show_coherence(corpus, dictionary, start=6, end=15):
    iter_num = []
    per_value = []
    coh_value = []
    
    for i in range(start, end+1):
        model = LdaModel(corpus=corpus, id2word=dictionary,
                         chunksize=1000, num_topics=i,
                         random_state=7)
        iter_num.append(i)
        pv = model.log_perplexity(corpus)
        per_value.append(pv)
        
        cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
        cv = cm.get_coherence()
        coh_value.append(cv)
        print(f'num_topics: {i}, perplexity: {pv:0.3f}, coherence: {cv:0.3f}')
        
    plt.plot(iter_num, per_value, 'g-')
    plt.xlabel('num_topics')
    plt.ylabel('perplexity')
    # plt.show()
    
    plt.plot(iter_num, coh_value, 'r--')
    plt.xlabel('num_topics')
    plt.ylabel('coherence')
    # plt.show()
    
    return per_value, coh_value


def main():
    filename = '/topic' + save_path[-1] + '_TopicModeling.html'
    df = pd.read_csv(data_path)
    papers = preprocess.extract_text(df)

    # token화
    texts = []
    orig_words = []
    for paper in papers:
        words, word_token = tokenizer(paper)
        orig_words.append(words)
        texts.append(word_token)

    # store stem result
    store_stem_result(texts, orig_words)

    # 토큰화 결과로부터 dictionary 생성
    dictionary = Dictionary(texts)
    print('Number of initial unique words in documents:', len(dictionary))

    # 문서 빈도수가 너무 적거나 높은 문서를 필터링하고 특성을 단어의 빈도 순으로 선택
    dictionary.filter_extremes(keep_n=2000, no_below=5, no_above=0.5)
    print("Number of unique words after removing rare and common words:", len(dictionary))
    print()

    # 카운트 벡터로 변환
    corpus = [dictionary.doc2bow(text) for text in texts]
    print(f'Number of unique tokens: {len(dictionary)}')
    print(f'Number of documents: {len(corpus)}')
    print()

    '''
    If you want to see the best number of topics based on the perplexity and coherence by topic, un-comment below.
    '''
    # perplexity 및 coherence 
    # per_value, coh_value = show_coherence(corpus, dictionary, start=6, end=15)

    # best_p_topic = per_value.index(min(per_value)) + 6
    # best_c_topic = coh_value.index(max(coh_value)) + 6
    # print("best topic num in perplexity:", best_p_topic)
    # print("best topic num in coherence:", best_c_topic)

    # choose num_topics according to the perplexity
    passes = 5
    model = LdaModel(corpus=corpus, id2word=dictionary, \
                        passes=passes, num_topics=num_topics, \
                        random_state=7)# LDA 모형을 pyLDAvis 객체에 전달
    lda_vis = gensimvis.prepare(model, corpus, dictionary, R=10)

    # save the result of predicted topic
    save_targets(model, corpus)
    
    # 각 토픽의 상위 비중 단어 확인
    print(model.print_topics(num_words=10))

    # html 저장
    pyLDAvis.save_html(lda_vis, './figure' + save_path + filename)

    

if __name__ == '__main__':
    global data_path, save_path, num_topics
    parser = argparse.ArgumentParser(description="-d input data path(csv) -s save path to store output -n number of topics")
    parser.add_argument('-d', help="input_data_path", required=True)
    parser.add_argument('-s', help="save_path", required=True)
    parser.add_argument('-n', help="num_topics", default=10)
    
    args = parser.parse_args()

    data_path = args.d; save_path = args.s; num_topics = args.n
    print("data_path:", data_path)
    print("save_path:", save_path)
    print("num_topics:", num_topics)
    print()

    main()