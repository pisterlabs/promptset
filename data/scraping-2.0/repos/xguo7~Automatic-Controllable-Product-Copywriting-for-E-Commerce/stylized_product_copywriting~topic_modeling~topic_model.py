import jieba
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import corpora, models, similarities
from gensim.models.ldamulticore import LdaMulticore
from gensim.test.utils import datapath
from gensim.test.utils import common_corpus, common_dictionary
from gensim.corpora import Dictionary
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



# ------------------------------对获取的数据进行分词-------------------------------
# 定义停词函数 对句子进行中文分词
def stopwordslist():
    stopwords = [line.strip() for line in open('./stop_words.txt', 'r', encoding='UTF-8').readlines()]
    stopwords.extend(['设计','#','穿','肌肤','效果','成分','面膜'])
    return stopwords

def seg_depart(sentence):
    sentence_depart = jieba.cut(sentence.strip())
    stopwords = stopwordslist()
    outstr = ''
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(ldamodel.get_document_topics(corpus)):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0: # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num),round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)    


sec_id='794'
#read data
path='/home/xiaojie.guo/daren_data/'+sec_id+'/'
with open(path+'daren'+sec_id+'_clean.csv','r', encoding = 'utf-8') as f:
    data=f.readlines()
text=[]
for item in data:
    text.append(item.split('|||')[2])


# 分词后的结果
result_fenci = []
for i in text:
        result_fenci.append(seg_depart(i).split(' ')[:-1])
#result_fenci = [i[1].split(' ')[:-1] for i in result_fenci]        

print('finish fenci!')
#重新训练模型并保存
id2word = corpora.Dictionary(result_fenci)
id2word.save_as_text(path+'lda_dic_'+sec_id)
print('dictionary saved!')


# 将字典转换为词袋,为文档中的每一个单词创建唯一的ID
corpus = [id2word.doc2bow(sentence) for sentence in result_fenci]   

'''
for i in [20,25]:#range(5, 20): 15:0.48
    lda_model = LdaMulticore(corpus, id2word=id2word, num_topics=i, workers=8)
    lda_model.save(datapath("model_lda_"+sec_id+"_"+str(i)+"class"))
    print('finish topic model training for '+str(i)+' classes!')
    print('Perplexity: ', lda_model.log_perplexity(corpus)) # a measure of how good the model is. lower the better.
    coherence_model_lda = CoherenceModel(model=lda_model, texts=result_fenci, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score: ', coherence_lda) # 主题一致性得分
    #更新新模型
    #lda_model = LdaMulticore.load(datapath('model_lda'))
    #corpus = [common_dictionary.doc2bow(t) for t in result_fenci]
    #lda_model.update(corpus)
'''
print('loading the saved LDA model!')
lda_model = LdaMulticore.load(datapath("model_lda_"+sec_id+"_"+str(16)+"class"))

#============assign the topic to each sentence
df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=result_fenci)
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
print('finished topic modeling assignment!')

#=============== see the topic and keywords
pd.options.display.max_colwidth = 100
sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
axis=0)
# Reset Index
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
sent_topics_sorteddf_mallet.to_excel(path+'topic_keywords_'+sec_id+'_16class.xlsx',index=False)
print('finished topic modeling inference!')


#df_dominant_topic[0:450000].to_excel(path+'assigned_topic_671_part1_10class.xlsx',index=False)
#df_dominant_topic[450000:].to_excel(path+'assigned_topic_671_part2_10class.xlsx',index=False)
df_dominant_topic.to_excel(path+'assigned_topic_'+sec_id+'_16class.xlsx',index=False)
