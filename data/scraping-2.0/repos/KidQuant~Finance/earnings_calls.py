import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

#Visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact, FloatRangeSlider

#spacy for language processing
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

#sklearn for feature extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

#gensim for topic models
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus

#topic modle viz
import pyLDAvis
from pyLDAvis.gensim import prepare

#evaluate parameter setting
import statsmodels.api as sm


plt.style.use('fivethirtyeight')
pyLDAvis.enable_notebook()
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.f}'.format

PROJECT_DIR = Path().cwd().parent.parent
earnings_path = PROJECT_DIR / '03_alternative_data' / '02_earnings_calls' / 'transcripts' / 'parsed'
experiment_path = Path('experiments')
clean_text = Path('data', 'clean_text.txt')

stop_words = set(pd.read_csv('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words', header = None, squeeze=True))

documents = []

for transcripts in earnings_path.iterdir():
    content = pd.read_csv(transcripts / 'content.csv')
    documents.extend(content.loc[(content.speaker !='Operator') & (content.content.str.len() > 5), 'content'].tolist())

len(documents)


word_count = pd.Series(documents).str.split().str.len()
ax = sns.distplot(np.log(word_count), kde =False)
ax.set_title('Log word count distribution')

word_count.describe(percentile=np.arange(.1,1.0,.1))
