# Topic modeling using vanilla latent dirichlet allocation

# Stdlib
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import re
from tabulate import tabulate

# Third party
import pandas as pd
import numpy as np
np.random.seed(2017)

from sklearn.manifold import TSNE

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
stop_en = stopwords.words('english')
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models import Phrases
from gensim.parsing.preprocessing import STOPWORDS
from stop_words import get_stop_words
from gensim import corpora, models
from gensim.models import CoherenceModel

import seaborn as sns
from bokeh.io import output_notebook
output_notebook()

from bokeh.plotting import figure, show
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider
from bokeh.layouts import column
from bokeh.palettes import all_palettes

import pyLDAvis.gensim
pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
%matplotlib inline


# Import data 
dataset = pd.read_csv('bt_1.csv').fillna('0') 

date = pd.to_datetime(dataset['D'],utc=True)
year = date.dt.year


'''Data preprocessing: transform the Message vector 
   9 times and apply regex to remove numerals'''

dataset['Message'] = dataset.Message.map(lambda x: re.sub(r'\d+', '', x))

# Convert words to lowercase
dataset['Message'] = dataset.Message.map(lambda x: x.lower())
print(dataset['Message'][0][:500])

# Tokenize the vector
dataset['Message'] = dataset.Message.map(lambda x: RegexpTokenizer(r'\w+').tokenize(x))
print(dataset['Message'][0][:25])

# Stem all words in the vector
snowball = SnowballStemmer("english")  
dataset['Message'] = dataset.Message.map(lambda x: [snowball.stem(token) for token in x])
print(dataset['Message'][0][:25])

# Lemmatize all words in vector
#from nltk.stem.wordnet import WordNetLemmatizer
#lemmatizer = WordNetLemmatizer()
#dataset['Message'] = dataset.Message.map(lambda x: [lemmatizer.lemmatize(token) for token in x])
#print(dataset['Message'][0][:25])


# Remove stopwords from vector
extra_stopwords = []
dataset['Message'] = dataset.Message.map(lambda x: [t for t in x if t not in stop_en])
dataset['Message'] = dataset.Message.map(lambda x: [t for t in x if not t in set(get_stop_words('english'))])
dataset['Message'] = dataset.Message.map(lambda x: [t for t in x if t not in STOPWORDS])
dataset['Message'] = dataset.Message.map(lambda x: [t for t in x if t not in extra_stopwords])
print(dataset['Message'][0][:25])

# Remove words with less than 2 characters
dataset['Message'] = dataset.Message.map(lambda x: [t for t in x if len(t) > 1])
print(dataset['Message'][0][:25])


# Puts the preprocessed text into an array
docs = np.array(dataset['Message']) # comment out array when running t-SNE


'''1. raw vector pulled from a dataframe = slow computation.
      I'm not sure what data type i should use to transform the 
      raw vector. Try an array'''

'''2. I put the transformed vector inside of an array
      and the data was able to pass through the bi/tri-gram for loop'''

# Add bigrams and trigrams that appear 
# more than 10 times to docs 

bigram = Phrases(docs, min_count=10)
trigram = Phrases(bigram[docs])
quadgram = Phrases(trigram[docs])
quingram = Phrases(quadgram[docs])
sexgram = Phrases(quingram[docs]) 


for idx in range(len(docs)):
   
    for token in bigram[docs[idx]]:
         
        if '_' in token:
         
            # Token is a bigram, add to document.
            docs[idx].append(token)
            
    for token in trigram[docs[idx]]:
      
        if '_' in token:
            
            # Token is a trigram, add to document.
            docs[idx].append(token)
#    for token in quadgram[docs[idx]]:
#        if '_' in token:
#            # Token is a quadgram, add to document.
#            docs[idx].append(token)   
#    for token in quingram[docs[idx]]:
#        if '_' in token:
#            # Token is a quindgram, add to document.
#            docs[idx].append(token)       
#    for token in sexgram[docs[idx]]:
#        if '_' in token:
#            # Token is a sexgram, add to document.
#            docs[idx].append(token)   


# Loads data into a new variable
texts = dataset['Message'].values
print('Total Number of documents: %d' % len(texts))

# Makes an index to word dictionary

dictionary = corpora.Dictionary(texts)
print('Number of unique tokens in initital documents:', len(dictionary))

# Filter words that occur in less than 20% of documents

dictionary.filter_extremes(no_below=10) 
print('Number of unique tokens after removing rare and common tokens:', len(dictionary))

# Converts the dictionary into a bag-of-words

corpus = [dictionary.doc2bow(text) for text in texts]


# Computes corpora coherence score  
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """

    coherence_values = []
    model_list = []
      
    for num_topics in range(start, limit, step):
      
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, 
                                                        texts=texts, start=2, limit=40, step=6)


# Coherence score plot
limit=40; start=2; step=6;
x = range(start, limit, step)
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# LDA model extracts topics from the corpus
ldamodel = models.ldamodel.LdaModel(corpus, id2word=dictionary, 
                                    num_topics=8, passes=100, 
                                    iterations=400, minimum_probability=0)

# prints coherence score
cm = CoherenceModel(model=ldamodel,texts=texts,dictionary=dictionary,coherence='c_v')
print(cm.get_coherence())

# prints topics
topics = ldamodel.print_topics(num_topics=30,num_words=6)
bt_1_topics = sorted(topics)
print (tabulate(bt_1_topics, floatfmt=".4f", headers=("#", 'topics'), tablefmt="fancy_grid"))


# LDA visualization tool
# Refactors the output of LDA model into a numpy matrix
refac_matrix = np.array([[y for (x,y) in ldamodel[corpus[i]]] for i in range(len(corpus))])

# Threshold filters out unconfident topics
threshold = 0.5
_idx = np.amax(refac_matrix, axis=1) > threshold  # idx of doc that above the threshold
refac_matrix = refac_matrix[_idx]

# Dimentionality reduction gives us a better plot
tsne = TSNE(random_state=2017, perplexity=30) # 5 30 50
tsne_embedding = tsne.fit_transform(refac_matrix)
tsne_embedding = pd.DataFrame(tsne_embedding, columns =['x','y'])
tsne_embedding['hue'] = refac_matrix.argmax(axis=1)

dataset['Date'] = pd.to_datetime(dataset.Date)


'''t-SNE scatter plot made with bokeh.
   You can move your mouse over a point
   to see specific words clustered in 
   their respective topics'''


source = ColumnDataSource(
        data=dict(
            x = tsne_embedding.x,
            y = tsne_embedding.y,
            colors = [all_palettes['Set1'][8][i] for i in tsne_embedding.hue],
            message = dataset.Message,
            year = year,
            alpha = [0.9] * tsne_embedding.shape[0],
            size = [7] * tsne_embedding.shape[0]))

hover_tsne = HoverTool(names=["df"], tooltips="""
    <div style="margin: 10">
        <div style="margin: 0 auto; width:300px;">
            <span style="font-size: 12px; font-weight: bold;">Message:</span>
            <span style="font-size: 12px">@message</span>
            <span style="font-size: 12px; font-weight: bold;">Year:</span>
            <span style="font-size: 12px">@year</span>
        </div>
    </div>
    """)
tools_tsne = [hover_tsne, 'pan', 'wheel_zoom', 'reset']
plot_tsne = figure(plot_width=700, plot_height=700, tools=tools_tsne, title='bt_1')
plot_tsne.circle('x', 'y', size='size', fill_color='colors', 
                 alpha='alpha', line_alpha=0, line_width=0.01, source=source, name="df")


callback = CustomJS(args=dict(source=source), code="""
    var data = source.data;
    var f = cb_obj.value
    x = data['x']
    y = data['y']
    colors = data['colors']
    alpha = data['alpha']
    message = data['message']
    year = data['year']
    size = data['size']
    for (i = 0; i < x.length; i++) {
        if (year[i] <= f) {
            alpha[i] = 0.9
            size[i] = 7
        } else {
            alpha[i] = 0.05
            size[i] = 4
        }
    }
    source.trigger('change');
""")

#slider = Slider(start=dataset.Date.min(), end=dataset.Date.max(), value=2016, step=1, title="Before year")
#slider.js_on_change('value', callback)

layout = column(plot_tsne) # slider
show(layout)
