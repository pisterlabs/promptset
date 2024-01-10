##Methods
    #Latent Semantic Anlalysis-LSA(Gizli Anlamsal Analiz)
    #Non Negative Matrix Factorization - NMF or NNMF( Negatif Olmayan Matriks Faktörizasyonu)
    #Latent Dirichlet Allocation - LDA(Gizli Dirichlet Ayırması)
    #Pachinko Allocation Model(Pachinko Dağılım Modeli)

#LDA-Demo
import numpy as np
import pandas as pd

news = pd.read_csv('turkish_news_70000.csv', index_col='id')

df = pd.DataFrame(news)

#print(df.columns)

##for LDA 'text' column will be used
news_dataset = pd.DataFrame(df[['text']])
#print(news_dataset.head())

'''
Main Text Cleaning Processes
1-All letters wil be lower
2-Punctuations will remove
3-Stopwords will remove(like adj.)
'''
import re
import string
import nltk
from nltk.corpus import stopwords

noktalama_isaretleri = string.punctuation
etkisiz_kelimeler = stopwords.words('turkish')
etkisiz_kelimeler.extend(["bir", "kadar", "sonra"])

def text_cleaning(txt):
    txt = txt.str.lower() #All letters will be lower
    txt = txt.replace("\\n"," ") #new line will replace with space
    txt = re.sub("’(\w+)","", str(txt)) #remove words that comes after apostrophe
    txt = re.sub("'(\w+)","", txt)
    txt = re.sub("[0-9]+","", txt) #remove numbers
    txt = "".join(list(map(lambda x:x if x not in noktalama_isaretleri else " ", txt))) # remove punctuations
    txt = " ".join([i for i in txt.split() if i not in etkisiz_kelimeler]) #remove some of stopwords
    txt = " ".join([i for i in txt.split() if len(i) > 1]) #remove an only letters

    return txt

#print(news_dataset.iloc[5].text)
'''
"Şubat ayında ihracat yüzde 3.7 arttı, ithalat yüzde 18.7 azaldı 04/03/2019 12:20\n
Ticaret Bakanı Ruhsar Pekcan şubat ayında ihracatın yüzde 3.7 artışla 14 milyar 312 milyon dolar, 
ithalatın yüzde 18.7 azalışla 16 milyar 161 milyon dolar olarak gerçekleştiğini açıkladı. 
Fotoğraf: Reuters\nBakan Pekcan şunları söyledi: “ Eskiden kullandığımız Özel Ticaret Sistemine 
göre de şubat ayında ihracatımız yüzde 3,5 artışla 13 milyar 603 milyon dolar olarak gerçekleşmiştir.
ÖTS’ye göre ithalatımız şubat ayında yüzde 16,6 düşüşle 15 milyar 793 milyon dolar seviyesinde gerçekleşmiştir. 
” Reklam"
'''

#cleaned_text = text_cleaning(news_dataset['text'].iloc[5])
#print(cleaned_text)
'''
şubat ayında ihracat yüzde arttı ithalat yüzde azaldı ticaret bakanı ruhsar pekcan şubat ayında 
ihracatın yüzde artışla milyar milyon dolar ithalatın yüzde azalışla milyar milyon dolar olarak 
gerçekleştiğini açıkladı fotoğraf reuters bakan pekcan şunları söyledi eskiden kullandığımız özel 
ticaret sistemine göre şubat ayında ihracatımız yüzde artışla milyar milyon dolar olarak gerçekleşmiştir
öts göre ithalatımız şubat ayında yüzde düşüşle milyar milyon dolar seviyesinde gerçekleşmiştir reklam
'''
news_dataset['cleaned_text'] = text_cleaning(news_dataset.loc[:,['text']].text)
#print(news_dataset.head(5))
news_dataset["cleaned_text_tokenization"] = news_dataset['cleaned_text'].apply(lambda x:x.split())

###tokenization
news_dataset["cleaned_text_tokenization"] = news_dataset['cleaned_text'].apply(lambda x:x.split())
#print(news_dataset.head(10))

import gensim
import pyLDAvis.gensim #LDA figurative show libs

#Kelime Listesi- Dictionary
tokenized_text = news_dataset["cleaned_text_tokenization"]
word_list = gensim.corpora.Dictionary(tokenized_text)

word_list.filter_extremes(no_below=1,no_above=0.7) #Filter

term_matrix = [word_list.doc2bow(term) for term in tokenized_text] #terms vectorization

#print('Number of unique tokens: %d' % len(word_list))
#print('Number of documents: %d' % len(term_matrix))

#LDA Model


# lda_model = gensim.models.ldamodel.LdaModel(corpus=term_matrix , num_topics=15,passes=10)

# subjects = lda_model.print_topics(num_words=7)

# for subject in subjects:
#     print(subject)

#Coherence Score(Tutarlılık)
from gensim.models import CoherenceModel

topic_number_range_list = range(9,30,3)
coherence_score_list = list()
topic_number_list = list()

for topic_number in topic_number_range_list:
    lda_model = gensim.models.ldamodel.LdaModel(corpus=term_matrix, id2word=word_list, num_topics=topic_number, passes=10)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_text, dictionary=word_list, coherence='c_v')
    coherence_score_lda = coherence_model_lda.get_coherence()
    coherence_score_list.append(coherence_score_lda)
    topic_number_list.append(topic_number)

import matplotlib.pyplot as plt

plt.plot(topic_number_list, coherence_score_list,"-"),
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")

plt.show()