# from wordcloud import WordCloud
# from matplotlib import pyplot as plt
# from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.test.utils import datapath
from nltk.stem import WordNetLemmatizer
# from pprint import pprint
from nltk.corpus import stopwords
# from datetime import date, timedelta

import requests
import gensim.corpora as corpora
import pandas as pd
import re
import gensim
import nltk
# import os

def run():

    d = "http://api:8000/all/"
    req = requests.get(d)

    j_data = req.json()

    if j_data != []:
        title = []
        text = []
        
        for i in j_data:
            title.append(i['title'])
            text.append(i['text'])

    df = pd.DataFrame({'text': text, 'title':title})
    #df = pd.read_csv('/home/alnaggar/PBL/data-1653249353296.csv')
    #df.columns = ['sk', 'url', 'title', 'text', 'tags', 'count', 'date', 'summary']

    print(df)
    news = df.copy()

    news['feature'] = news['title'] + ' ' + news['text']
    #Removing punctuation from the text column
    news['feature'] = news['feature'].map(lambda x: re.sub('[,\.!?]', '', x))
    # Convert the titles to lowercase
    news['feature'] = news['feature'].map(lambda x: x.lower())
    #news = news.drop(columns = ['title','url','tags','count', 'summary', 'date'])

    news = news.fillna('')




    #Displaying most repeated words in the text through wordclouds

    # Join the different processed documents together.
    long_string = ','.join(list(news['feature'].values))

    # Create a WordCloud object
    # wordc = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')


    # Visualize and generate word cloud
    #plt.imshow(wordc.generate(long_string))

    #plt.axis("off") 
    #plt.show()

    #image = wordc.to_image()
    #image.show()



    lem = WordNetLemmatizer()

    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))



    data = news.feature.values.tolist()
    data_words = list(sent_to_words(data))

    # remove stop words

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    #print(corpus[:1][0][:30])


    # number of topics
    num_topics = 10
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
    # Print the Keyword in the topics
    #pprint(lda_model.print_topics())
    #print(lda_model.get_document_topics(corpus[0]))
    #doc_lda = lda_model[corpus]

    # Save model to disk.
    temp_file = datapath("./lda_model.model")
    lda_model.save(temp_file)



    #Uncomment this to see topics analysis
    """"
    #Visulaizing the keywords and probability distribution
    import pyLDAvis
    import pyLDAvis.gensim_models
    import pickle 
    today = date.today()


    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)

    LDAvis_data_filepath = os.path.join('/home/alnaggar/Documents/PBL Data/davis/'+str(num_topics) + today.strftime("%y-%m-%d"))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, '/media/alnaggar/F47C61617C611F9A/PBL Data/davis/'+ str(num_topics) + today.strftime("%y-%m-%d") +'.html')
    LDAvis_prepared

    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts = data_words, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    """