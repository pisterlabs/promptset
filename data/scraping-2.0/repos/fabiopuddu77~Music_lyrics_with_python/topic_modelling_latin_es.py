
from collections import Counter
from PIL import Image
from gensim.models import CoherenceModel
from wordcloud import WordCloud, STOPWORDS
import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
from pprint import pprint
from preprocessing_es import *
from topic_modelling_genre import format_topics_sentences, visualize_topics, show_topic_clusters_gen
from topic_modelling_genre import wordcloud_topic

if __name__ == '__main__':

    # lettura dataset
    df = pd.read_csv("dataset/Dataset.csv", error_bad_lines=False, sep=',')

    '''Latin'''
    df = df.loc[(df['Lingua'] == 'es') & (df['Genere'] == 'Latin')]

    ##### PROVA DF PIU PICCOLO ######
    #df = df[0:100]

    data_classes = ['Latin']
    n_topics = 2
    testo = df['Testo']
    data_ready = Pulizia_es(testo)

    ''' Scelta del genere di cui vogliamo fare la topic analysis, si sono selezionati i 3 
    generi più frequenti nel dataset'''

    # Create Dictionary
    id2word = corpora.Dictionary(data_ready)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]

    # Build LDA model

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=n_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=10,
                                                alpha='symmetric',
                                                iterations=100,
                                                per_word_topics=True)

    pprint(lda_model.print_topics())
    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()

    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    #df_dominant_topic.to_csv('df_dominant_topic_es.csv')

    # Mostrami a quale topic appartiene il documento numero 1
    df1 =  df_dominant_topic.loc[(df_dominant_topic['Document_No'] == 1)].Document_No
    print('Il documento numero: ',df1.to_string(index=False))

    df2 = df_dominant_topic.loc[(df_dominant_topic['Document_No'] == 1)].Dominant_Topic
    print('Appartiene al topic ', df2.to_string(index=False))

    #plot_document(df_dominant_topic)
    wordcloud_topic(lda_model)
    #key_words(lda_model)
    #sentences_chart(lda_model, corpus)


    ''' Visualize HTML reports of topics and topic clusters by genre '''
    visualize_topics(lda_model, corpus, nome=data_classes[0])
    show_topic_clusters_gen(lda_model, corpus, nome=data_classes[0], n_topics=n_topics)