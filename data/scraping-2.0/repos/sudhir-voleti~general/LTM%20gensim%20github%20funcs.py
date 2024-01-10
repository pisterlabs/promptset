# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:00:28 2020

@author: 20052
"""
# setup chunk
import re, time
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim setup
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# import gensim.parsing.preprocessing
from gensim.parsing.preprocessing import strip_punctuation, strip_tags, strip_numeric
from nltk.stem.wordnet import WordNetLemmatizer   
from nltk.corpus import stopwords
import string  # for the .join() func
import matplotlib.pyplot as plt


## --- randomly sample k1 sents from n1 docs from a raw corpus
import random
from random import sample
from nltk.tokenize import sent_tokenize

# --- util func to sample k1 sents from a doc
def _sampl_sent(doc0, k1=1):
    sents0 = sent_tokenize(doc0)
    test_range0 = [x for x in range(len(sents0))]
    k2 = min(k1, len(sents0))
    sent_index0 = sample(test_range0, k2); sent_index0
    sent_samples0 = [sents0[x] for x in sent_index0]; sent_samples0
    return sent_samples0

## -- wrapper func around unit func
def sampl_sents_series(series0, pkey_series0, n1=20, k1=5):
	test_range = [x for x in range(len(series0))]
	test_samples = sample(test_range, n1)   # sampling w/o replacemt
	a0 = series0.iloc[test_samples]   # subsetted df
	#pkey_series1 = pkey_series0.iloc[test_samples]

	df_out1 = pd.DataFrame(columns=['prim_key', 'sents'])

	for ind_num in test_samples:
		doc0 = series0.iloc[ind_num]
		if str(doc0)=='nan':
			continue
		else:
			sents_samples0 = _sampl_sent(series0.iloc[ind_num], k1=k1)
			pkey0 = pkey_series0.iloc[ind_num]; pkey0
			df_out0 = pd.DataFrame({'prim_key':[pkey0]*len(sents_samples0), 'sents':sents_samples0})
			df_out1 = pd.concat([df_out1, df_out0], axis=0)

	return df_out1

# test-drive above
# path_mktg = 'D:/Earning Call prepRmks & QnA df/write_files/mktg wordlist based/'
# %time out_df0 = pd.read_csv(path_mktg + 'out_df0.csv'); out_df0.columns # 11s 
# %time df_qna_out1 = sampl_sents_series(out_df0.nonmo_sents_qna, out_df0.prim_key, n1=1000, k1=5) #2.4s


# for dtm processing
from sklearn.feature_extraction.text import CountVectorizer

lemma = WordNetLemmatizer()
stop_words = stopwords.words('english')

## routine 1 - textclean per doc
def textClean(corpus_raw, stop_words0 = stopwords.words('english')):
    text1 = [strip_punctuation(str(doc)) for doc in corpus_raw]
    text1 = [strip_tags(doc) for doc in text1]
    text1 = [strip_numeric(doc) for doc in text1]
    text1 = [[" ".join([i for i in doc.lower().split() if i not in stop_words0])] for doc in text1]
    text2 = [[word for word in ' '.join(doc).split()] for doc in text1]
    normalized = [[" ".join([lemma.lemmatize(word) for word in ' '.join(doc).split()])] for doc in text1]
    return normalized

## routine 2 - gridsearch on coherence vals
def compute_coherence_values1(dictionary, corpus, texts, num_topics_list):
    coherence_values = []
    model_list = []
    for num_topics in num_topics_list:
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100,
                                           update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence()); print(num_topics)
	
    return model_list, coherence_values   

## routine 2a - plot coherence metrics. input is outp of prev func
import matplotlib.pyplot as plt
def plot_coherence(coherence_values, num_topics_list):

	coher = [(coherence_values[i0], num_topics_list[i0]) for i0 in range(len(num_topics_list))]
	opt_num_topics = [y for (x,y) in coher if x == max(coherence_values)]

	x = num_topics_list
	plt.plot(x, coherence_values)
	plt.xlabel("Num Topics")
	plt.ylabel("Coherence score")
	plt.axvline(x = opt_num_topics, color='r', label='opt_numTopics')
	plt.legend(("coherence_values"), loc='best')
	plt.show()

## routine 3 - gridsearch via perplexity scores
def compute_perplexity_values(model_list, corpus, num_topics_list):
	perplexity_values = []
	for i0 in range(len(num_topics_list)):
		num_topics = num_topics_list[i0] 
		model = model_list[i0]
		perplexity_values.append(model.log_perplexity(corpus))	        
	return perplexity_values  # note, list of 1 obj only returned

## routine 3a - plot perplexity metrics
def plot_perplexity(perplexity_values, num_topics_list):

	perpl = [(perplexity_values[i0], num_topics_list[i0]) for i0 in range(len(num_topics_list))]
	opt_num_topics = [y for (x,y) in perpl if x == min(perplexity_values)]

	plt.plot(num_topics_list, perplexity_values)
	plt.xlabel("Num Topics")
	plt.ylabel("Perplexity score")
	plt.legend(("perplexity_values"), loc='best')
	plt.axvline(x = opt_num_topics, color='r')
	plt.show()
    
## routine 4 - get factor matrices
def build_beta_df(lda_model, id2word):  # lda_model is the optimal_model here
    beta = lda_model.get_topics()  # shape (num_topics, vocabulary_size).
    beta_df = pd.DataFrame(data=beta)

    # convert colnames in beta_df 2 tokens
    token2col = list(id2word.token2id)
    beta_df.columns = token2col
    # beta_df.loc[0,:].sum()  # checking if rows sum to 1

    # convert rownames too, eh? Using format(), .shape[] and range()
    rowNames=['topic' + format(x+1, '02d') for x in range(beta_df.shape[0])]
    rowNames_series = pd.Series(rowNames)
    beta_df.rename(index=rowNames_series, inplace=True)
    return(beta_df)

## routine 4a - get gamma matrix
def build_gamma_df(lda_model, corpus_raw, id2word):  # lda_model is the optimal_model here
    gamma_doc = []  # empty list 2 populate with gamma colms
    num_topics = lda_model.get_topics().shape[0]
    
    for doc in range(len(corpus_raw)):
        doc1 = str(corpus_raw.iloc[doc]).split()
        bow_doc = id2word.doc2bow(doc1)
        gamma_doc0 = [0]*num_topics  # define list of zeroes num_topics long
        gamma_doc1 = lda_model.get_document_topics(bow_doc)
        gamma_doc2_x = [x for (x,y) in gamma_doc1]#; gamma_doc2_x
        gamma_doc2_y = [y for (x,y) in gamma_doc1]#; gamma_doc2_y
        for i in range(len(gamma_doc1)):
            x = gamma_doc2_x[i]
            y = gamma_doc2_y[i]
            gamma_doc0[x] = y  # wasn't geting this in list comprehension somehow 
        gamma_doc.append(gamma_doc0)
        
    gamma_df = pd.DataFrame(data=gamma_doc)  # shape=num_docs x num_topics
    topicNames=['topic' + format(x+1, '02d') for x in range(num_topics)]
    topicNames_series = pd.Series(topicNames)
    gamma_df.rename(columns=topicNames_series, inplace=True)
    return(gamma_df)    

## routine 5 - get dominant Topic DF
def domi_topic_df(gamma_df, optimal_model):
	row0 = gamma_df.values.tolist()
	row=[]
	for i in range(len(row0)):
		row1 = list(enumerate(row0[i]))
		row1_y = [y for (x,y) in row1]
		max_propn = sorted(row1_y, reverse=True)[0]
		row2 = [(i, x, y) for (x, y) in row1 if y==max_propn]
		row.append(row2)

	sent_topics_df = pd.DataFrame(columns = ['Doc_num', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])
	for row1 in row:
		for (doc_num, topic_num, prop_topic) in row1:
			wp = optimal_model.show_topic(topic_num)
			topic_keywords = ", ".join([word for word, prop in wp])
			sent_topics_df0 = pd.DataFrame({'Doc_num': doc_num, 'Dominant_Topic': topic_num, 
							'Perc_Contribution': prop_topic, 'Topic_Keywords':topic_keywords}, index=[0])
			sent_topics_df = pd.concat([sent_topics_df, sent_topics_df0], ignore_index=True) 
                                                       
    
	#sent_topics_df.columns = ['Doc_num', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
	return(sent_topics_df)    

## Routine 6 - processing raw data
def build_gensim_corpus(corpus_raw, stop_words0):
    corpus_cleaned = textClean(corpus_raw, stop_words0)  # corpus cleaned of html tags, puncs, lemmas
    corpus_tokenized = [[word for word in ' '.join(doc).split()] for doc in corpus_cleaned]  # word_tokenize first
    id2word = corpora.Dictionary(corpus_tokenized)  # Create Dictionary from word_tokenized corpus
    corpus_gensim = [id2word.doc2bow(text) for text in corpus_tokenized]  # Building gensim corpus. TF DTM creation.
    return(corpus_cleaned, corpus_tokenized, id2word, corpus_gensim)


## Routine 7 - get factor matrices using optimal K
def ltm_outp_df(model_list, num_topics_list, id2word, K):
    K1 = K  - num_topics_list[0]; K1   # account for starting point offset
    optimal_model = model_list[K1]
    # model_topics = optimal_model.show_topics(formatted=False)

    beta_df = build_beta_df(optimal_model, id2word)  # 0.004 secs
    beta_df = beta_df.T

    gamma_df = build_gamma_df(optimal_model, corpus_cleaned); gamma_df.shape # gamma_df.iloc[:8,:8]
    sent_topics_df = domi_topic_df(gamma_df)  # 2.64 secs

    return(beta_df, gamma_df, sent_topics_df)

## Routine 8 - wrapper over all above funcs
def ltm_wrapper(corpus_raw, num_topics_list, stop_words0):  # start1, limit1, step1
    
    corpus_cleaned, corpus_tokenized, id2word, corpus_gensim = build_gensim_corpus(corpus_raw, stop_words0)	
    print("build_gensim_corpus done.\n")
    # num_topics_list = [x for x in range(start1, limit1, step1)]; num_topics_list
    corpus_cleaned_series = pd.Series(corpus_cleaned)
    model_list, coherence_values = compute_coherence_values1(id2word, corpus_gensim, corpus_tokenized, num_topics_list)    
       
    perplexity_values = compute_perplexity_values(model_list, corpus_gensim, num_topics_list)
    
    print("grid searches done.\n")
    
    # print gridSearch results
    coher = [(coherence_values[i0], num_topics_list[i0]) for i0 in range(len(num_topics_list))]
    perpl = [(perplexity_values[i0], num_topics_list[i0]) for i0 in range(len(num_topics_list))]
    opt_num_topics_coher = [y for (x,y) in coher if x == max(coherence_values)]; opt_num_topics_coher[0]
    opt_num_topics_perpl = [y for (x,y) in perpl if x == min(perplexity_values)]; opt_num_topics_perpl[0]
	
    # display plots		
    plot_coherence(coherence_values, num_topics_list)
    plot_perplexity(perplexity_values, num_topics_list)
		
    print("opt_num_topics_coher: ", opt_num_topics_coher[0])
    print("opt_num_topics_perpl: ", opt_num_topics_perpl[0])
    
    K = opt_num_topics_coher[0]; print("optimal num_topix: ", K,"\n")  # default
    K10 = [x for x in range(len(num_topics_list)) if num_topics_list[x] == opt_num_topics_coher[0]]
    K1 = K10[0]; print("K1: ", K1, "\n")   # account for starting point offset
    optimal_model = model_list[K1]
    
    beta_df = build_beta_df(optimal_model, id2word)  # 0.004 secs
    beta_df = beta_df.T; beta_df.shape
    tokens = list(beta_df.index)
    beta_df.insert(0, "tokens", tokens)  # insert tokens as a colm	
    
    gamma_df = build_gamma_df(optimal_model, corpus_cleaned_series, id2word); gamma_df.shape 
    sent_topics_df = domi_topic_df(gamma_df, optimal_model)  
    print("factor matrices done.\n")
    
    return(beta_df, gamma_df, sent_topics_df, optimal_model)

## Routine 8b: single topic LDA run from raw corpus
def single_lda_run(corpus_raw, num_topics):
    stop_words0 = stopwords.words('english')
    corpus_cleaned, corpus_tokenized, id2word, corpus_gensim = build_gensim_corpus(corpus_raw, stop_words0) 
    

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_gensim, id2word=id2word, num_topics=num_topics, random_state=100,
                                               update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    # lda_model.show_topics()
    
    
    beta_df = build_beta_df(lda_model, id2word)
    tokens = beta_df.columns.to_list(); type(tokens)
    beta_df = beta_df.T
    beta_df.insert(0, "tokens", tokens); beta_df.shape    
    gamma_df = build_gamma_df(lda_model, corpus_raw, id2word) 
    
    corpus_series = pd.Series(corpus_tokenized)
    for i0 in range(len(corpus_tokenized)):
        a0 = ' '.join(corpus_tokenized[i0]); a0
        corpus_series.iloc[i0] = a0
        if i0%1000 == 0:
            print(i0)  # 24 secs    
    
    dtm_ml_pr = list2dtm(corpus_series.tolist(), 0.90, 0.01)   # 15 secs    
    dtm_select1, beta_df_select, beta_df_logi1 = get_dtm_beta(dtm_ml_pr, beta_df)

    return(dtm_select1, beta_df_select, beta_df_logi1, gamma_df, beta_df)

# test-drive and incorp into gitsource
# num_topics = 8
# dtm_select1, beta_df_select, beta_df_logi1, gamma_df, beta_df = single_lda_run(corpus_raw, num_topics)

# simpler version of above
def single_lda_run_simple(corpus_raw, num_topics):
    stop_words0 = stopwords.words('english')
    corpus_cleaned, corpus_tokenized, id2word, corpus_gensim = build_gensim_corpus(corpus_raw, stop_words0) # 16m
    

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_gensim, id2word=id2word, num_topics=num_topics, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True) # 48m
    # lda_model.show_topics()
    
    
    beta_df = build_beta_df(lda_model, id2word) # 0.02s
    tokens = beta_df.columns.to_list(); type(tokens)
    beta_df = beta_df.T
    beta_df.insert(0, "tokens", tokens); beta_df.shape    
    gamma_df = build_gamma_df(lda_model, corpus_raw, id2word) # 2m 50s
    
    corpus_series = pd.Series(corpus_tokenized)
    for i0 in range(len(corpus_tokenized)):
        a0 = ' '.join(corpus_tokenized[i0]); a0
        corpus_series.iloc[i0] = a0
        if i0%10000 == 0:
            print(i0)  # 24 secs    
    
    return(gamma_df, beta_df)

# util unit func to zero-fy all entries in a row except max one
def _hard_allocate_row(row):
	row1 = row.apply(lambda x:round(x,6)); row1
	row1a = (row1 == row1.max())*row1.max(); row1a
	return(row1a)

def build_shell_beta_df(beta_df, top_n=50):
	n1 = beta_df.shape[1]
	beta_df1a = beta_df.iloc[:,1:(n1+1)]; beta_df1a

	#%time beta_df1 = beta_df1a.apply(lambda x: _hard_allocate_row(x), axis=0) # didn't work
	beta_df0 = beta_df
	for i0 in range(beta_df.shape[0]):
		n1 = beta_df.shape[1]
		row = beta_df.iloc[i0, 1:(n1+1)]
		row1 = _hard_allocate_row(row)
		beta_df0.iloc[i0, 1:(n1+1)] = row1
		if i0%5000==0:
			print(i0)

	shell_df = pd.DataFrame({'term_no':[(x+1) for x in range(top_n)]}); shell_df
	for i0 in range(1, beta_df.shape[1]):    
		sub_beta_df = beta_df.iloc[:,[0, i0]]; sub_beta_df
		colm_name = sub_beta_df.columns[(sub_beta_df.shape[1]-1)]; colm_name
		a00 = sub_beta_df.sort_values(by=colm_name, ascending=False); a00
		a0 = a00.iloc[:top_n,0].tolist(); a0
		shell_df.insert(shell_df.shape[1], colm_name, a0)

	return(shell_df, beta_df0) 

# test-drive
#%time shell_df, beta_df0 = hard_allocate_beta_df(beta_df)  # takes time in minutes coz of for loop

## now define func to yield revised gamma matrix based on shell_df
def revise_gamma_by_topic(corpus_raw, shell_df, gamma_df):
	n1 = shell_df.shape[1]; n1 # ncol of shell_df
	gamma_df1 = gamma_df0 # create copy

	select_tokens = []
	for i0 in range(1, n1):	
		a0 = shell_df.iloc[:, i0].tolist(); a0[:5]
		select_tokens.append(a0)

	# building dtm of corpus_raw
	tf_vect = CountVectorizer(lowercase=True, min_df=5, ngram_range=(1,1))
	dtm_tf = tf_vect.fit_transform(corpus_raw); print(dtm_tf.shape) # 15s
	dtm_tf1 = dtm_tf.todense(); print(dtm_tf1.shape) # 4s
	dtm_toks = tf_vect.get_feature_names()

	# build empty panda to populate
	for i0 in range(len(select_tokens)):
		index0 = pd.Series([x for x in range(dtm_tf.shape[0])])
		dtm_tf_df = pd.DataFrame(index=index0, columns=select_tokens[i0]); dtm_tf_df.shape

		# find where all a topic's tokens are in dtm_tf
		zeroes_colm = pd.Series([0]*dtm_tf.shape[0]); zeroes_colm[:8]
		for i1 in range(len(select_tokens[i0])):
			a1 = np.where(np.asarray(dtm_toks) == select_tokens[i0][i1])[0].tolist(); #a1[0]
			if len(a1) > 0:
				dtm_tf_df.iloc[:,i1] = dtm_tf1[:,a1[0]] #colm0
			else:        
				dtm_tf_df.iloc[:,i1] = zeroes_colm

		dtm_tf_df.iloc[:8,:8]
		a1 = dtm_tf_df.sum(axis=1); a1[:8]
		gamma_df1.iloc[:,i0] = gamma_df0.iloc[:,i0] * a1; gamma_df1.iloc[:8,i0]

	return(gamma_df1)

# test-drive
#%time gamma_df1 = revise_gamma_by_topic(corpus_raw, shell_df, gamma_df0) # 39s

## Routine 9 - Build and display wordclouds
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def make_wordcloud(beta_sub_df):  # only 2 colms - tokens and topic
	mean0 = beta_sub_df.iloc[:,1].mean(); mean0
	series0 = round(beta_sub_df.iloc[:,1]/mean0, 0); series0[:8]
	beta_sub_df.iloc[:,1] = series0

	# build dict as wordcl input
	d = {}
	for a, x in beta_sub_df.values:
		d[a] = x

	# invoke wordcloud
	wordcloud = WordCloud()
	wordcloud.generate_from_frequencies(frequencies=d)
	plt.figure(figsize=(20,10))
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	title0 = beta_sub_df.columns[1]
	plt.title(title0)
	plt.show()
	
## Routine 10 - build DTMs for COGs later in R
def list2dtm(text_list, max_thresh, min_thresh):  # bottom 2.5% tokens by TF dropped. intermed routine.
    
    text = text_list
    vectorizer = CountVectorizer(lowercase=False, max_df=max_thresh, min_df=min_thresh)  # from sklearn.feature_extraction.text import CountVectorizer
    # vectorizer.fit(text)  # tokenize and build vocab
    vector = vectorizer.fit_transform(text)  # encode document
    
    # build DTM outp as DF
    a0 = vector.toarray()   # dense matrix form
    a1 = np.sum(a0, axis = 0)  # vec obj of colm sums
    a2 = vectorizer.vocabulary_  # dict obj
    a3 = {k: v for k, v in sorted(a2.items(), key=lambda item: item[1])}  # sort keys by value
    a4 = [k for (k, v) in a3.items()]  # list of tokens
    dtm = pd.DataFrame(data = a0, columns = a4)
    return(dtm)

# wrapper over intermed routine abv
def series2dtm(corpus_raw, max_thresh, min_thresh):  
	corpus_cleaned = textClean(corpus_raw)
	sents_str = []
	for i0 in range(len(corpus_cleaned)):
		a0 = str(corpus_cleaned[i0]).strip('[]'); a0
		a1 = re.sub(r"\'","",a0); a1
		sents_str.append(str(a1)); sents_str

	dtm = list2dtm(sents_str, max_thresh, min_thresh)
	return(dtm)

# subset a dtm based on a wordlist match
def subset_dtm(dtm0, wordlist0):
	colname_list0 = list(dtm0.columns)
	a1 = [x for x in range(len(colname_list0)) if type(colname_list0[x]) != str] # anomaly correction
	for i0 in a1:
		colname_list0[i0] = str(colname_list0[i0])
   
	colnames = " ".join(colname_list0)
	ind_list = []

	for i0 in range(len(wordlist0)):
		a0 = re.findall(wordlist0[i0], colnames); a0	
		if len(a0) > 0:
			a1 = [x for x in range(dtm0.shape[1]) if len(re.findall(wordlist0[i0], colname_list0[x]))>0]; a1
			ind_list.extend(a1)

	# build sub_dtm, find summaries, bind into df and save
	sub_dtm = dtm0.iloc[:, ind_list]; sub_dtm.shape # <0.2 s
	return(sub_dtm)

# test-driving above
# %time sub_dtm_qna = subset_dtm(df_qna_test, wl_mktg) # 7.73s

## save dtm as sparse mat with colm names as list
from scipy.sparse import csr_matrix, save_npz, load_npz
def save_dtm_sparsely(dtm0, path0, dtm_name = 'dtm_name'):
	dtm_cols = dtm0.columns
	a0 = csr_matrix(dtm0.values) # 8m 38s
	save_npz(path0 + dtm_name + '.npz', a0) # 6.7s
	pd.Series(dtm_cols.tolist()).to_csv(path0 + dtm_name + "_colms.csv") #<1s

# test-drive
# %time save_dtm_sparsely(sub_dtm_pr, path0, dtm_name = 'subdtm_pr_mktg') # 3.07s


# Routine 11: to get DTM & beta_df for COG & Wordcl
def get_dtm_beta(dtm_select, beta_df):

	## reshape beta_df & dtm into colms found in one another
	a0 = dtm_select.columns; a0[:8]
	a00 = beta_df['tokens'].to_list(); a00[:8]

	a1 = [bool(x in a0) for x in a00]; a1[:8]
	# a1.count(True)  # 1119
	beta_df_select = beta_df.loc[a1,:]; beta_df_select.shape
	beta_df_select.columns

	a2 = [bool(x in a00) for x in a0]; a2[:8]
	# a2.count(True)
	dtm_select1 = dtm_select.loc[:, a2]; dtm_select1.shape

	# routine 2 extract dtm_sub for particular topic
	beta_df_logi1 = beta_df_select
	beta_df_maxVals = []

	# Go colm by colm and extract maxVal for each row
	for i1 in range(beta_df_select.shape[0]):
		a0 = max(beta_df_select.iloc[i1, 1:]); a0
		beta_df_maxVals.append(a0)

	# eval each colm for maxVal incidence
	for i2 in range(1, beta_df_select.shape[1], 1):
		logi1 = [bool(beta_df_select.iloc[x, i2] == beta_df_maxVals[x]) for x in range(beta_df_select.shape[0])]
		beta_df_logi1.iloc[:, i2] = logi1
   	
	return dtm_select1, beta_df_select, beta_df_logi1
