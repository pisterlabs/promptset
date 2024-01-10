import streamlit as st
state = st.session_state
from streamlit import components
import pandas as pd
from datetime import datetime
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# import colorlover as cl
# colors = cl.to_rgb(cl.scales['7']['qual']['Set2'])
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# @st.cache_resource
def get_ta_models(data):
    # Build the bigram model
    bigram = gensim.models.Phrases(data, min_count=3, threshold=50) # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    data_bigrams = make_bigrams([bigram_mod[doc] for doc in data])
    # Create Dictionary
    id2word = corpora.Dictionary(data_bigrams)
    # Create Corpus
    texts = data_bigrams
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    return data_bigrams, id2word, corpus

def get_lda_model(id2word, corpus, num_topics):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,
            num_topics=num_topics, random_state=100, update_every=1,
            chunksize=200, passes=5, alpha='auto', per_word_topics=True)

    return lda_model

# @st.cache_data
def get_topic_df(df_filtered, _lda_model, corpus):
    # make df of top topics
    topic_df = pd.DataFrame()

    for i,r in df_filtered.iterrows():

        try:
            tops = _lda_model.get_document_topics(corpus[i])
            td = {t[0]:t[1] for t in tops}

            td['top_topic'] = [str(k) for k,v in td.items() if v == max([v for k,v in td.items()])][0]
            td['docid'] = str(i)
            td['date'] = r['cleandate']
            # topic_df = topic_df.append(td, ignore_index=True)
            topic_df = pd.concat([topic_df, pd.DataFrame([td])])
        except:
            pass

    return topic_df

def get_wc(lda_model, i):
    weighted = {kwd[0]:kwd[1] for kwd in lda_model.show_topic(i, topn=100)}
    wordcloud = WordCloud(background_color="white", colormap='twilight').generate_from_frequencies(weighted)

    wc = plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.close()
    return wc.figure

# distribution of topics over time
def topics_by_month(lda_model, topic_df, ta_abs_btn):

    # build plot
    fig = go.Figure()
    t_df = pd.DataFrame()

    for n,g in topic_df.groupby(topic_df.date):
        data = {'month':n}
        if ta_abs_btn == 'Absolute':
            for c in g[[c for c in g.columns if str(c).isdigit()]]:
                data[c] = g[c].agg(sum)
                # data[c] = g[c].apply(lambda x: x.sum())

                sizemin=4
        else:
            for c in g[[c for c in g.columns if str(c).isdigit()]]:
                data[c] = g[c].agg(sum) / len(g)
                sizemin=6
        # t_df = t_df.append(data, ignore_index=True)
        t_df = pd.concat([t_df, pd.DataFrame([data])])

    t_df = t_df.sort_values('month')
    maxval = t_df[[c for c in t_df.columns if str(c).isdigit()]].max(axis=1).max(axis=0)

    for topic_num in t_df[list(set([c for c in t_df.columns if str(c).isdigit()]))]:

        maxsize = t_df[topic_num].apply(lambda x: x/maxval*100)
        kwds = ', '.join([lda_model.id2word[t[0]] for t in lda_model.get_topic_terms(topic_num)])
        fig.add_trace(go.Scatter(x=t_df.month.tolist(), y=t_df[topic_num],
                mode='markers',marker=dict(
                                 size=maxsize,
                                 sizemin=sizemin),
                                 name=f'Topic {topic_num + 1} - {kwds}',
                                 line_shape='spline'))
        fig.update_layout(legend=dict(yanchor="top", y=-0.1, xanchor="left",
                        x=0, itemsizing='constant', traceorder='normal'))
        fig.update_layout(height=700)

    return fig

def plot_coherence(coherence_df):

    coherence_df['Number of topics'] = coherence_df['Number of topics'].astype(int)
    coherence_df.set_index('Number of topics', inplace=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]], x_title='Number of topics')

    fig.add_trace(go.Scatter(x=coherence_df.index, y=coherence_df['Coherence'],
                            mode='lines+markers', connectgaps=True,name='Coherence',
                            line_shape='spline'))
    fig.update_yaxes(title_text="Coherence", secondary_y=False, showgrid=False)

    fig.add_trace(go.Scatter(x=coherence_df.index, y=coherence_df['Perplexity'],
                            mode='lines+markers', connectgaps=True, name='Perplexity', line_shape='spline'), secondary_y=True)
    fig.update_yaxes(title_text="Perplexity", secondary_y=True, showgrid=False)

    return fig
