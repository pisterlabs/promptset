#%%
## 01_데이터셋 입력
import pandas as pd
import numpy as np

rawdata = pd.read_csv('rawdata_abstract.csv')
documents = pd.DataFrame(rawdata)

len(documents)

#%%
## 02_데이터 전처리 A
# 데이터 전처리 함수 import
import re # 문자열 정규 표현식 패키지
import nltk
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import preprocess_string

nltk.download('stopwords')

def clean_text(d):
    pattern = r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', d)
    return text

def clean_stopword(d):
    stop_words = stopwords.words('english')
    stop_words.extend(['geographic', 'surface', 'line', 'point', 'high', 'paper', 'fractal', 'japan', 'africa', 'eros', 'multi', 'dimension', 'relation', 'multi', 'time'])
    return ' '.join([w.lower() for w in d.split() if w.lower() not in stop_words and len(w) > 3])

def preprocessing(d):
    return preprocess_string(d)

#%%
## 02_데이터 전처리 B
# 용어 치환
from tqdm import tqdm # 작업 프로세스 시각화

replace_list = pd.read_csv('replace_list.csv')

def replace_word(review):
    for i in range(len(replace_list['before_replacement'])):
        try:
            # 치환할 단어가 있는 경우에만 용어 치환 수행
            if replace_list['before_replacement'][i] in review:
                review = review.replace(replace_list['before_replacement'])[i], replace_list(['after_replacement'][i])
        except Exception as e:
            print(f"Error 발생 / 에러명: {e}")
    return review

documents['keword_prep'] = ''
review_replaced_list = []
for review in tqdm(documents['keword']):
    review_replaced = replace_word(str(review)) # 문자열 데이터 변환
    review_replaced_list.append(review_replaced)

documents['keword_prep'] = review_replaced_list
documents.head()
len(documents)


# %%
## 02_데이터 전처리 C
# 특수문자, 숫자 등 불용어 제거
# 토근화 및 리스토로 변경
documents['keword_prep'] = documents['keword_prep'].apply(clean_stopword)
tokenized_docs = documents['keword_prep'].apply(preprocessing)
tokenized_docs = tokenized_docs.to_list()

# %%
## 02_데이터 전처리 D
# 토큰의 개수가 1보다 작은 것을 삭제
import numpy as np
drop_docs = [index for index, sentence in enumerate(tokenized_docs) if len(sentence) <=5]
docs_texts = np.delete(tokenized_docs, drop_docs, axis = 0)
len(docs_texts)

# %%
## 03_LDA 토픽 모델링 A
from gensim import corpora
from gensim.models import LdaModel

dictionary = corpora.Dictionary(docs_texts)
corpus = [dictionary.doc2bow(text) for text in docs_texts]

lda_model = LdaModel(corpus, num_topics = 10, id2word=dictionary) # 10개 괜찮은 듯
topics = lda_model.print_topics()
topics
# pd.DataFrame(topics, columns='test')
# topics.to_csv("houseprice.csv")

# %%
## 03_LDA 토픽 모델링 B
# 최적 주제 선정을 위한 코드
from gensim.models.coherencemodel import CoherenceModel

min_topics, max_topics = 5, 15
coherence_scores = []

for num_topics in range(min_topics, max_topics):
    model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
    coherence = CoherenceModel(model=model, texts=docs_texts, dictionary=dictionary)
    coherence_scores.append(coherence.get_coherence())
    
coherence_scores
# %%
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')

x = [int(i) for i in range(min_topics, max_topics)]

plt.figure(figsize=(10, 6))
plt.plot(x, coherence_scores)
plt.xlabel('number of topics')
plt.ylabel('coherence_scores')
plt.show()

#%%
## 03_LDA 토픽 모델링 C
# 토픽 주제 출력
from gensim.models import LdaModel

lda_model = LdaModel(corpus, num_topics = 7, id2word=dictionary) # 5 -> 7/ 30 -> 8
topics = lda_model.print_topics()
topics

# %%
## 04_LDA 토픽 모델링 D 시각화
import pyLDAvis.gensim_models

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)
# %%

## 참고문헌
# https://heytech.tistory.com/401

#%%
test = pd.DataFrame(docs_texts)
test.to_csv('test222222222.csv')

pd.DataFrame(topics, columns='test')
topics.to_csv("houseprice.csv")