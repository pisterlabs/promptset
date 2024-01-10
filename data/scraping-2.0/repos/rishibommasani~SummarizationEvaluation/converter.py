import random
import torch
import math
# from newsroom import jsonl
import jsonlines as jsonl
from fragments import Fragments
# from newsroom.analyze.rouge import ROUGE_N
# from newsroom.analyze.rouge import ROUGE_L
import pickle
import spacy
import nltk
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm
spacy_tokenizer = spacy.load("en_core_web_sm")
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.wrappers import LdaMallet
path_to_mallet_binary = "Mallet/bin/mallet"
from transformers import BertTokenizer, BertForNextSentencePrediction
import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from dataset import *
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
nlp = spacy.load('en', disable=['parser', 'ner'])
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok
detokenizer = Detok()

def detok(tokens):
	text = detokenizer.detokenize(tokens)
	text = re.sub('\s*,\s*', ', ', text)
	text = re.sub('\s*\.\s*', '. ', text)
	text = re.sub('\s*\?\s*', '? ', text)
	return text


def cmu2us(data):
	print("Converting dataset to our format")
	output = []
	for example in tqdm(data):
		document = "".join([detok(s) for s in example.src_sents])
		summary = " ".join([" ".join(s) for s in example.tgt_sents])
		f = Fragments(summary, document)
		output.append({'summary' : summary, 'text' : document, 'coverage' : f.coverage(), 'density' : f.density(), 'compression' : f.compression()})
	print("Converted dataset to our format")
	return output


def us2cmu(data):
	print("Converting dataset to CMU format")
	output = []
	for example in tqdm(data):
		document = [s.split(" ") for s in sent_tokenize(example['text'])]
		summary = [s.split(" ") for s in sent_tokenize(example['summary'])]
		output.append((document, summary))
	print("Converted dataset to CMU format")
	return output
