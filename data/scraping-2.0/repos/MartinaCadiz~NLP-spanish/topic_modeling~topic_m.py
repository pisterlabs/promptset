from collections import Counter
import seaborn as sns
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim
import pickle 
import os
from matplotlib.patches import Rectangle

import numpy as np
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
import matplotlib.colors as mcolors
from bokeh.io import output_notebook
from bokeh.plotting import figure, show, output_notebook, save#, output_file
from bokeh.models import HoverTool, value, LabelSet, Legend, ColumnDataSource
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
import gensim

def word_count_weight_keywords(model, dataset_cleaned, size_figure=(24,10)):
    topics = model.show_topics(formatted=False)
    data_flat = [w for w_list in dataset_cleaned for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    if len(model.show_topics())==1:
      fig,ax = plt.subplots(len(model.show_topics()), figsize=size_figure, sharey=True)
      ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==0, :], color=cols[0], width=0.5, alpha=0.3, label='Word Count')
      ax_twin = ax.twinx()
      ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==0, :], color=cols[0], width=0.2, label='Weights')
      ax.set_ylabel('Word Count', color=cols[0])
      ax.set_title('Topic: ' + str(0), color=cols[0], fontsize=16)
      ax.tick_params(axis='y', left=False)
      ax.set_xticklabels(df.loc[df.topic_id==0, 'word'], rotation=30, horizontalalignment= 'right')
      ax.legend(loc='upper right'); ax_twin.legend(loc='center right')  
    else:
      # Plot Word Count and Weights of Topic Keywords
      fig, axes = plt.subplots(len(model.show_topics()), figsize=size_figure, sharey=True)
      for i, ax in enumerate(axes.flatten()):
          ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
          ax_twin = ax.twinx()
          ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
          ax.set_ylabel('Word Count', color=cols[i])
          ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
          ax.tick_params(axis='y', left=False)
          ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
          ax.legend(loc='upper right'); ax_twin.legend(loc='center right')
    fig.tight_layout(w_pad=2)    
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
    plt.show()
    return df


def LDA_visual(lda_model,words,corpus,id2word):
    # Visualize the topics
    pyLDAvis.enable_notebook()

    LDAvis_data_filepath = os.path.join('/content/ldavis_tuned_')
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word,R=words)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)

    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)

    pyLDAvis.save_html(LDAvis_prepared, '/ldavis_tuned_.html')

    return LDAvis_prepared


def top_vocab(model):
    top_words_per_topic = []
    for t in range(model.num_topics):
        top_words_per_topic.extend([(t,) + x for x in model.show_topic(t)])
    return pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P'])

def format_topics_sentences(corpus, texts, model=None):
    sent_topics_df = pd.DataFrame()
    for i, row_list in enumerate(model[corpus]):
        row = sorted(row_list, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords','Text']
    return(sent_topics_df)

def most_representative_sent(model_,corpus_,texts_):
    df_topic_sents_keywords = format_topics_sentences(model=model_, corpus=corpus_, texts=texts_)
  
    df = pd.DataFrame()
    for i, grp in df_topic_sents_keywords.groupby('Dominant_Topic'):
        df = pd.concat([df, grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],axis=0) 
    df.reset_index(drop=True, inplace=True)
    df.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
    return df



def sentences_chart(corpus, id2word, num_topics,_model='LDA', start = 0, end = 13):
    corp = corpus[start:end]
    mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    if _model =='LDA':
      model =  gensim.models.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=num_topics, per_word_topics=True)
    fig, axes = plt.subplots(end-start, 1, figsize=(20, (end-start)*0.95), dpi=160)       
    axes[0].axis('off')
    for i, ax in enumerate(axes):
        if i > 0:
            corp_cur = corp[i-1] 
            topic_percs, wordid_topics, wordid_phivalues = model[corp_cur]
            word_dominanttopic = [(model.id2word[wd], topic[0]) for wd, topic in wordid_topics]    
            ax.text(0.0001, 0.5, "Doc " + str(i-1) + ": ", verticalalignment='center',
                    fontsize=13, color='black', transform=ax.transAxes, fontweight=700)

            # Draw Rectange
            topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
            ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1, 
                                   color=mycolors[topic_percs_sorted[0][0]], linewidth=1))

            word_pos = 0.06
            for j, (word, topics) in enumerate(word_dominanttopic):
                if j < 14:
                    ax.text(word_pos, 0.5, word,
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=10, color=mycolors[topics],
                            transform=ax.transAxes, fontweight=700)
                    word_pos += .006 * len(word)  # to move the word for the next iter
                    ax.axis('off')
            ax.text(word_pos, 0.5, ' ',
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=16, color='black',
                    transform=ax.transAxes)       

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end-2), fontsize=22, y=0.95, fontweight=700)
    plt.tight_layout()
    plt.show()


def tsne_plot(model_,corp,name_model,text,interactive_labels = True):
# Get topic weights
    topic_weights = []
    for i, row_list in enumerate(model_[corp]):
        topic_weights.append([w for i, w in row_list])

    # Array of topic weights    
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    # Plot the Topic Clusters using Bokeh
    output_notebook()
    n_topics = len(model_.show_topics())
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    if interactive_labels is False:
        plot = figure(title="t-SNE Clustering of {} {} Topics".format(n_topics,name_model), plot_width=900, plot_height=700)
        plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
        show(plot)
    else:
        df_dominant_topic = format_topics_sentences(model=model_, corpus=corp, texts=text)
        source = ColumnDataSource(dict(
        x=tsne_lda[:,0],
        y=tsne_lda[:,1],
        color=mycolors[topic_num],
        label=df_dominant_topic['Dominant_Topic'].apply(lambda s: 'Belongs to topic '+str(s)),
        topic_key= df_dominant_topic['Dominant_Topic'],
        title= df_dominant_topic['Text'],
        content = df_dominant_topic['Topic_Keywords']    
        ))
        title = "t-SNE Clustering of {} LDA Topics".format(n_topics)

        plot_lda = figure(plot_width=1500, plot_height=600,
                            title=title, tools="pan,wheel_zoom,box_zoom,reset,hover")

        plot_lda.scatter(x='x', y='y', source=source,legend_field='label',
                        color='color', alpha=0.8, size=5)#'msize', )

        # hover tools
        hover = plot_lda.select(dict(type=HoverTool))
        hover.tooltips = [
            ("Sentence", "@title"),
            ("KeyWords", "@content"),
            ("Topic", "@topic_key"),
        ]

        plot_lda.legend.location = "top_left"

        output_file('scatter_topics.html')
        show(plot_lda)




def compute_coherence_values(dictionary, corpus, texts, id2word, limit, start=2, step=3, model_ ='LDA'):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    model_ : model type
    Returns:
    -------
    model_list : List of LDA/LSI topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model =  gensim.models.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=num_topics)
        if model_ == 'LSI':
          model =  gensim.models.LsiModel(corpus=corpus,id2word=id2word,num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())


    plt.plot(range(start, limit, step), coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.title('Best number of topics with '+model_+' model')
    plt.legend(("coherence values"), loc='best')
    plt.show()

    return model_list, coherence_values