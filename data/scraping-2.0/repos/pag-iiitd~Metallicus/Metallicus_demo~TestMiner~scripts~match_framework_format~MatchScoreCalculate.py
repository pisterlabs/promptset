import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import gensim
from gensim import corpora,models
import pickle
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import random
import re	
import sys
from gensim.models import CoherenceModel
from clus_simmat import *
from nltk.stem.wordnet import WordNetLemmatizer

def key_with_max_val(d):
	val = list(d.values())
	k = list(d.keys())
	#if not val:
	#	print(d)
	max_val = max(val)
	return k[val.index(max_val)], max_val


def _adjust_param_sim_score(matching_pairs_score, score_thresh=0.8, w_high_score_pairs=0.2):
	matched_pairs = []
	high_score_pairs = []
	score = 0.0
	for param in matching_pairs_score:
		if matching_pairs_score[param]:
			m_param, sim_score = key_with_max_val(matching_pairs_score[param])
			matched_pairs.append(tuple([param, m_param, sim_score]))
			score += (1-w_high_score_pairs)*sim_score/len(matching_pairs_score)
			if sim_score > score_thresh:
				high_score_pairs.append(tuple([m_param, sim_score]))
	if high_score_pairs:
		score += w_high_score_pairs*sum(h_score[-1] for h_score in high_score_pairs)/len(high_score_pairs)
	elif matched_pairs:
		score += w_high_score_pairs*sum(m_score[-1] for m_score in matched_pairs)/len(matched_pairs)
	return score, matched_pairs


def calculate_params_distance(q_params, m_params,similarity_matrix,dictionary):
	sim_score = 0.0
	matching_pairs_score = {}
	"""Param Weights are calculated with a strategy as below.
	All the matches are weighted based on the similarity score.
	For example, if sim(qp1, mp2)> 0.8 weight of the match is readjusted 
	from others' weights to give it higher weight""" 
	
	if True:#len(q_params) >= len(m_params):
		for q_p_key in q_params:
			matching_pairs_score[q_p_key] = {}
			for m_p_key in m_params:
				if q_params[q_p_key] and m_params[m_p_key]:
					q_doc = '{} {}'.format(' '.join(camel_case_split2(q_p_key)).lower(), q_params[q_p_key])
					m_doc = '{} {}'.format(' '.join(camel_case_split2(m_p_key)).lower(), m_params[m_p_key])
					matching_pairs_score[q_p_key][m_p_key] = similarity_matrix.inner_product(dictionary.doc2bow([lemmatizer.stem(word) for word in q_doc.split()]), dictionary.doc2bow([lemmatizer.stem(word) for word in m_doc.split()]), normalized=True)
	"""elif len(q_params) < len(m_params):
		for m_p_key in m_params:
			matching_pairs_score[m_p_key] = {}
			for q_p_key in q_params:
				if q_params[q_p_key] and m_params[m_p_key]:
					q_doc = '{} {}'.format(q_p_key, q_params[q_p_key])
					m_doc = '{} {}'.format(m_p_key, m_params[m_p_key])
					matching_pairs_score[m_p_key][q_p_key] = similarity_matrix.inner_product(dictionary.doc2bow([lemmatizer.stem(word) for word in q_doc.split()]), dictionary.doc2bow([lemmatizer.stem(word) for word in m_doc.split()]), normalized=True)
	"""
	if matching_pairs_score:
		sim_score, matched_pairs = _adjust_param_sim_score(matching_pairs_score)		
	return sim_score, matched_pairs


def calculate_doc_distance(qdoc, qsig, mdoc, msig, similarity_matrix, dictionary):

	q_fn = get_cleaned_doc(get_function_name(qsig))
	m_fn = get_cleaned_doc(get_function_name(msig))
	if m_fn==q_fn:
		fsim=1
	else:
		fsim = similarity_matrix.inner_product(dictionary.doc2bow([lemmatizer.stem(word) for word in q_fn.split()]), dictionary.doc2bow([lemmatizer.stem(word) for word in m_fn.split()]), normalized=True)
	#print(qsig, msig)
	
	if not qdoc or not mdoc:
		sim = 0
		return (0,0,0)
	q_params = get_param_docs_dict(qdoc, qsig)
	m_params = get_param_docs_dict(mdoc, msig)
	q_doc = get_cleaned_doc(qdoc)
	m_doc = get_cleaned_doc(mdoc)
	#print(q_params)
	sim = similarity_matrix.inner_product(dictionary.doc2bow([lemmatizer.stem(word) for word in q_doc.split()]), dictionary.doc2bow([lemmatizer.stem(word) for word in m_doc.split()]), normalized=True)
	psim = 0
	if q_params and m_params:
		psim, matched_pairs = calculate_params_distance(q_params, m_params,similarity_matrix, dictionary)
		#print(matched_pairs)
		"""except:
			print(q_params)
			print(m_params)"""

	return (sim,psim,fsim)

def get_function_name(fn_sig):
	fn_sig = fn_sig[:fn_sig.find('(')]
	return ' '.join(' '.join(camel_case_split2(fn_sig.split()[-1])).lower().split('_')).strip()


def _calculate_adaptive_sim(doc_sim, p_sim, threshold=0.5):
	if doc_sim < threshold:
		return 0.7*doc_sim + 0.3*p_sim
	return 0.5*doc_sim + 0.5*p_sim


def calculate_combined_sim(doc_sim, p_sim, adaptive=False):
	if not p_sim:
		return doc_sim
	if adaptive:
		return _calculate_adaptive_sim(doc_sim, p_sim)
	return 0.5*doc_sim + 0.5*p_sim
