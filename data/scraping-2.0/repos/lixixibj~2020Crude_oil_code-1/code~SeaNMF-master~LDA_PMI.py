# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 22:19:01 2020

@author: nihao
"""
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

def get_train(path1):
    f = open(path1)
    #读取全部内容
    lines = f.readlines()  #lines在这里是一个list
    return lines 

def main():
    docs = get_train('D:/ByResearch/基于文本的原油油价预测/20200615code/code/SeaNMF-master/data/wedata.txt')
    docs = [s.strip().split() for s in docs]
    
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=10, no_above=0.2)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
        
    # Make a index to word dictionary.
    temp = dictionary[0]  # only to "load" the dictionary.
    id2word = dictionary.id2token
    
    PMI = []
    for i in range(2,11):   
        print(i)
        lda_model = LdaModel(corpus=corpus, id2word=id2word,
                             iterations=100, num_topics=i)
        # Print the Keyword in the 5 topics
        print(lda_model.print_topics())
        
        coherence_model_lda = CoherenceModel(model=lda_model, texts=docs, dictionary=dictionary, coherence='c_uci')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        del lda_model
        PMI.append(coherence_lda)
    print(PMI)

if __name__ == '__main__':
    main()

    PMI_LDA = [-1.1249,-1.3806,-2.3789,-1.6854,-2.6971,-2.2079,-2.5428,-3.1219,-3.8324]
    PMI_NMF = [0.1124,0.3624,0.6837,0.5339,0.6112,0.2780,0.5090,0.4507,0.4745]
    
    plt.figure(figsize=(6, 4))
    plt.grid(c='grey',ls='--')
    plt.plot(range(2,11),PMI_LDA,'blue',label='PMI_LDA')
    plt.plot(range(2,11),PMI_NMF,'black',label='PMI_seaNMF')
    plt.title('Fluctuation of PMI score with the number of topics')
    plt.xlabel('Num of topics (k)')
    plt.ylabel('PMI')
    plt.legend()
    plt.rcParams['savefig.dpi'] = 2000
    plt.show()      














