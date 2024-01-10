# -*- coding: utf-8 -*-
import xlrd
import string
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
#import genism
from gensim import corpora, models
from gensim.models import CoherenceModel
import sys, errno
import xlsxwriter

dictionary = corpora.Dictionary.load("D:\\aproject\\dictionary1.dic")
#print( dictionary)
corpus = corpora.MmCorpus("D:\\aproject\\corpus1.m")
#print(len(corpus))
# 使用 gensim 来创建 LDA 模型对象
Lda = models.ldamodel.LdaModel
# 在 DT 矩阵上运行和训练 LDA 模型# iterations=50
Perplexity_list=[]
coherence_list=[]
#Passes=[300]
Iterations=[25]
for j in range(20,30):
    print('num_topics:',j)
    for it in Iterations:
        print('num_iter:',it)
        ldamodel = Lda(corpus, num_topics=j, id2word = dictionary,passes=300,alpha='auto',eta='auto',iterations=25, random_state=123,minimum_probability=0.0)
        #print(ldamodel.print_topics(num_topics=12, num_words=30))
        for i in  ldamodel.show_topics(num_topics=j):
            print (i[0], i[1])
        #, random_state=1  , num_words=10
        print('-----------------')

        file='D:\\aproject\\audience_text_classifier\\text_classifier_'+str(j)+'_i'+str(it)+'_v1.xlsx'
        workbook= xlsxwriter.Workbook(file)
        worksheet= workbook.add_worksheet(u'sheet1')
        for i in range(0,len(brand)):
            index1=0
            index2=0
            index1=brand0.index(brand[i])
            index2=i+360
            classifier1=ldamodel[corpus[index1]]
            ll1=[]
            for a,b in classifier1:
                ll1.append((b,a))
            ll1.sort(reverse=True)
            
            classifier2=ldamodel[corpus[index2]]
            ll2=[]
            for a,b in classifier2:
                ll2.append((b,a))
            ll2.sort(reverse=True)

            worksheet.write(i,0,brand[i])
            index3=0
            for a,b in ll1:
                if(index3<3):
                    if(a>0):
                        worksheet.write(i,index3+1,b)
                index3+=1 
            
            worksheet.write(i,4,influencer[i])
            index4=0
            for a,b in ll2:
                if(index4<3):
                    if(a>0):
                        worksheet.write(i,index4+5,b)
                    else:
                        worksheet.write(i,index4+5,-1)
                index4+=1
        workbook.close()
        print('----------save-------')
        
        # Compute Perplexity
        print('\nPerplexity: ', ldamodel.log_perplexity(corpus))  
        Perplexity_list.append(ldamodel.log_perplexity(corpus))
        # a measure of how good the model is. lower the better.
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=ldamodel,corpus=corpus,dictionary=dictionary, coherence='u_mass')
        try:
            coherence_lda = coherence_model_lda.get_coherence()
        except IOError as e:
            if e.errno == errno.EPIPE:
                print('err')
        print('\nCoherence Score: ', coherence_lda)
        coherence_list.append(coherence_lda)
        '''
        file2='D:\\aproject\\text_classifier\\text_classifier_vector.xlsx'
        workbook2= xlsxwriter.Workbook(file2)
        worksheet2= workbook2.add_worksheet(u'sheet1')
        for kk in range(0,len(corpus)):
            index_k=0
            if(kk<360):
                worksheet2.write(kk,0,brand0[kk])
            else:
                worksheet2.write(kk,0,influencer[kk-360])
            for a,b in ldamodel[corpus[kk]]:
                
                worksheet2.write(kk,1+index_k*2,a)
                worksheet2.write(kk,2+index_k*2,b)
                index_k+=1    
                    
        workbook2.close()
        '''
print(Perplexity_list)
print(coherence_list)




