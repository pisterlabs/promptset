# gensim
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath

# utils
from datetime import datetime
import logging
import re
import os
import sys
from tqdm import tqdm

# TSNE dependencies
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.colors as mcolors
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# dashboards
import pyLDAvis

# In order to deal with python version import in Kaggle
if sys.version_info == (3,9,13, 'final', 0):
    import pyLDAvis.gensim_models
else:
    import pyLDAvis.gensim


# utils
def parse_logfile(path_log):
    matcher = re.compile(r'(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity')
    likelihoods = []
    with open(path_log) as source:
        for line in source:
            match = matcher.search(line)
            if match:
                likelihoods.append(float(match.group(1)))
    return likelihoods


class LDATopicModeling():
    
    def __init__(self, df,
                 lang_preprocess,
                 decade = [1960],
                 directory = "/kaggle/working/models/",
                 gensim_log = "/kaggle/working/gensim.log",
                 existing = False,
                 n_topics = 10,
                 worker_nodes = None,
                 chunks = 10000,
                grid_search = False,
                epochs = 30,
                metric = 'c_v',
                eval_every = 2):
        
        # filter metric
        if (metric != 'c_v') and (metric != 'u_mass'):
            raise Exception('{} is not known metric'.format(metric))

        # get the metat data
        self.__meta_data = df[df['decade'].isin(decade)][['artist', 'title']]
        
        # Apply preprocessing on decade data
        self.__documents = df.loc[df['decade'].isin(decade), 'lyrics'].apply(lang_preprocess)
            
        # Create a corpus from a list of texts
        self.__id2word = Dictionary(self.__documents.tolist())
        self.__corpus = [self.__id2word.doc2bow(doc) for doc in self.__documents.tolist()]
        
        #training
        if os.path.isfile(existing):
            # Load a potentially pretrained model from disk.
            self.model = LdaModel.load(directory)
            self.__cv_results = None # no grid_search
            self.__n_topics = n_topics
        elif not grid_search:
            self.model = LdaMulticore(
                corpus=tqdm(self.__corpus),
                id2word=self.__id2word,
                num_topics=n_topics,
                workers=worker_nodes,
                passes=epochs,
                chunksize=chunks,
                eval_every = eval_every)
            self.__likelihood = parse_logfile(gensim_log)
            self.__n_topics = n_topics
            self.__cv_results = None
        else: # cross validation
            
            # hyperparameter
            alpha = []#np.arange(0.01, 1, 0.3).tolist()
            alpha.append('symmetric')
            alpha.append('asymmetric')
            
            # hyperparameter
            eta = []#np.arange(0.01, 1, 0.3).tolist()
            eta.append('symmetric')
            
            # compute results of the cross_validation
            cv_results = {
                 'topics': [],
                 'alpha': [],
                 'eta': [],
                 'c_v': [],
                 'u_mass': []
            }
            
            # topic range
            topic_range = range(5, n_topics+1)
            
            # prevent the computation time
            total=(len(eta)*len(alpha)*len(topic_range))
            print("total lda computation: ",total)
            
            # initialize incremental list
            model_list = []
            likelihood_list = []
            
            grid = [{'n_top': k, 'alpha': a, 'eta' : e}
                        for k in topic_range for a in alpha for e in eta]

            for param in grid:           
                # train the model
                model = LdaMulticore(
                    corpus=self.__corpus,
                    id2word=self.__id2word,
                    num_topics=param['n_top'],
                    workers=worker_nodes,
                    passes=epochs,
                    alpha=param['alpha'],
                    eta=param['eta'],
                    eval_every = eval_every,
                    chunksize=chunks)
                
                # track likelihood
                likelihood = parse_logfile(gensim_log)
                likelihood_list.append(likelihood)
                
                # compute coherence
                cv = CoherenceModel(
                    model=model,
                    texts=self.__documents,
                    dictionary=self.__id2word,
                    coherence='c_v')
                
                # compute coherence
                umass = CoherenceModel(
                    model=model,
                    texts=self.__documents,
                    dictionary=self.__id2word,
                    coherence='u_mass')
                
                print('coherence cv:{}, coherence umass:{}\nalpha:{}\neta:{}\ntopic:{}'.format(
                    cv.get_coherence(), umass.get_coherence(), param['alpha'], param['eta'], param['n_top']))
                
                # Save the model results
                cv_results['topics'].append(param['n_top'])
                cv_results['alpha'].append(param['alpha'])
                cv_results['eta'].append(param['eta'])
                cv_results['c_v'].append(cv.get_coherence())
                cv_results['u_mass'].append(umass.get_coherence())
                model_list.append(model)
            # retrieve index of the highest coherence model

            best_index = np.argmax(cv_results[metric])
            
            # choose the model given the best coherence
            self.model = model_list[best_index]
            
            # choose likelihood of the best model
            self.__likelihood = likelihood_list[best_index]
            
            # save results as attribute
            self.__cv_results = cv_results
            
            self.__n_topics = cv_results['topics'][best_index]
                                    
        # directory path
        self.__directory = directory
        
        # decade
        self.__decade = decade

        # c_v or u_mass
        self.__metric = metric
        
    # getters
    @property
    def get_id2word(self):
        return self.__id2word
    
    @property
    def get_corpus(self):
        return self.__corpus
    
    @property
    def get_likelihood(self):
        return self.__likelihood
    
    @property
    def get_cv_results(self):
        return pd.DataFrame(self.__cv_results) if self.__cv_results else None
    
    def plot_coherence(self):
        """plot coherence given the two metrics
        """
        if self.__cv_results is None:
            raise Exception('No cross validation available')
        
        # get the dataframe
        df_res = self.get_cv_results
                       
        # groupby by metric
        # create the layout
        fig = go.Figure()
        for metric in ['c_v', 'u_mass']:
            df_grp = df_res.groupby(['topics'])[metric].max().reset_index()
            fig.add_trace(
                go.Scatter(
                    x=df_grp['topics'],
                    y=df_grp[metric],
                    mode='lines+markers',
                    name=metric
                )
            )
        fig.update_layout(
            title = "coherence over topics by",
            xaxis_title="topic",
            yaxis_title="coherence")
        return fig
                                    
                                    
    def save_current_model(self):
        # retrieve time
        now = datetime.now()
        # create the directory if it doesn't exist
        try:
            os.makedirs(self.__directory)
        except:
            pass
        
        self.model.save(self.__directory + 'lda_' + now.strftime("%d%m%Y_%H%M%S"))

    def get_perplexity(self):
        return self.model.log_perplexity(self.__corpus)
    
    def get_coherence(self):
        coherence_model_lda = CoherenceModel(
            model=self.model,
            texts=self.__documents,
            dictionary=self.__id2word,
            coherence='c_v')
        return coherence_model_lda.get_coherence()
    
    
    # data vizualisation
    def dashboard_LDAvis(self):
        
        # enable notebook mode
        pyLDAvis.enable_notebook()

        # use the conresponding function given the
        # suitable imported module
        if sys.version_info == (3,9,13, 'final', 0):
            vis = pyLDAvis.gensim_models.prepare(
                self.model, self.__corpus,
                dictionary = self.model.id2word
            )
        else:
            vis = pyLDAvis.gensim.prepare(
                self.model, self.__corpus,
                dictionary = self.model.id2word
            )
        return vis
        
    def plot_tsne(self, components = 2):
        # n-1 rows each is a vector with i-1 posisitons, where n the number of documents
        # i the topic number and tmp[i] = probability of topic i
        topic_weights = []
        for row_list in self.model[self.get_corpus]:
            tmp = np.zeros(self.__n_topics)
            for i, w in row_list:
                tmp[i] = w
            topic_weights.append(tmp)

        # Array of topic weights    
        arr = pd.DataFrame(topic_weights).fillna(0).values
        
        # Keep the well separated points
        # filter documents with highest topic probability given lower bown (optional)
        # arr = arr[np.amax(arr, axis=1) > 0.35]
      
        # Dominant topic number in each doc (to compute color)
        topic_num = np.argmax(arr, axis=1)

        # tSNE Dimension Reduction
        tsne_model = TSNE(n_components=components, verbose=1, init='pca')
        tsne_lda = tsne_model.fit_transform(arr)
        
        #mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    

        # components
        if components == 2:
            # create representation df
            tsne_rep = pd.DataFrame({'x': tsne_lda[:,0],
                                     'y': tsne_lda[:,1],
                                     'topic': topic_num,
                                     'artist': self.__meta_data['artist'],
                                     'title': self.__meta_data['title']})
            sample_text = tsne_rep.sample(frac=0.01)
            # create figure
            fig = go.Figure()
            # create a topic list
            topic_list = sorted(tsne_rep['topic'].unique().tolist())
            # cross over topics
            for topic in topic_list:
                # select rows of the given topic
                df_topic = tsne_rep[tsne_rep['topic'] == topic]
                # add scatter plot
                fig.add_scatter(
                    x=df_topic['x'],
                    y=df_topic['y'],
                    mode='markers',
                    name='Topic '+ str(topic),
                    customdata=np.stack(
                        (df_topic['title'],
                        df_topic['artist']),
                        axis=-1
                    ),
                    hovertemplate = 
                        "title: %{customdata[0]}<br>" +
                        "artist: %{customdata[1]}<br>" +
                        "x: %{x}" + "y: %{y}"
                )
            # add random text
            fig.add_scatter(
                x=sample_text['x'],
                y=sample_text['y'],
                mode='text',
                text=sample_text['artist'],
                customdata=np.stack(
                        (sample_text['title'],
                        sample_text['artist']),
                        axis=-1
                ),
                textposition="bottom center",
                hovertemplate = 
                    "title: %{customdata[0]}<br>" +
                    "artist: %{customdata[1]}<br>" +
                    "x: %{x}" + "y: %{y}",
                showlegend=False
            )
            fig.update_layout(
                title = "t-SNE 2d Clustering of {} LDA Topics ({})" \
                    .format(self.__n_topics, self.__decade),
                xaxis_title="x",
                yaxis_title="y",
                autosize=False,
                width=900,
                height=700
            )
        elif components == 3:
            # create representation df
            tsne_rep = pd.DataFrame({'x': tsne_lda[:,0],
                                     'y': tsne_lda[:,1],
                                     'z': tsne_lda[:,2],
                                     'topic': topic_num,
                                     'artist': self.__meta_data['artist'],
                                     'title': self.__meta_data['title']})
            sample_text = tsne_rep.sample(frac=0.001)
            # create figure
            fig = go.Figure()
            # create a topic list
            topic_list = sorted(tsne_rep['topic'].unique().tolist())
            # cross over topics
            for topic in topic_list:
                # select rows of the given topic
                df_topic = tsne_rep[tsne_rep['topic'] == topic]
                # add scatter plot
                fig.add_scatter3d(
                    x=df_topic['x'],
                    y=df_topic['y'],
                    z=df_topic['z'],
                    mode='markers',
                    name='Topic '+ str(topic),
                    customdata=np.stack(
                        (df_topic['title'],
                        df_topic['artist']),
                        axis=-1
                    ),
                    hovertemplate = 
                        "title: %{customdata[0]}<br>" +
                        "artist: %{customdata[1]}<br>" +
                        "x: %{x}" + "y: %{y}" + "z: %{z}"
                )
            # add random text
            fig.add_scatter3d(
                x=sample_text['x'],
                y=sample_text['y'],
                z=sample_text['z'],
                mode='text',
                text=sample_text['artist'],
                customdata=np.stack(
                        (sample_text['title'],
                        sample_text['artist']),
                        axis=-1
                ),
                hovertemplate = 
                    "title: %{customdata[0]}<br>" +
                    "artist: %{customdata[1]}<br>" +
                    "x: %{x}" + "y: %{y}" + "z: %{z}",
                showlegend=False
            )
            fig.update_layout(
                title = "t-SNE 3d Clustering of {} LDA Topics ({})" \
                    .format(self.__n_topics, self.__decade),
                xaxis_title="x",
                yaxis_title="y")
        else:
            raise Exception("Components exceed covered numbers : {}".format(components))
        return fig
        
    
    def plot_likelihood(self, upper_bound=35):
        fig = go.Figure(
            go.Scatter(x=[i for i in range(0,upper_bound)], y=self.__likelihood[-upper_bound:],
                       mode='lines',
                       name='likelihood'))
        fig.update_layout(
            title = "Likelihood over passes",
            xaxis_title="passes",
            yaxis_title="Likekihood")
        return fig
    
    def dashboard(self, w=1000, h=1200):
        fig = make_subplots(
            rows=3, cols=2,
            specs=[[{"colspan": 2}, None],
                   [{}, {"rowspan": 2}],
                   [{}, None]],
           subplot_titles=['tsne 2d', 'coherence scores (c_v)', 'likelihood','coherence scores (u_mass)'],
           row_heights=[0.7, 0.15, 0.15],
           horizontal_spacing = 0.05,
           vertical_spacing = 0.05
        )
        fig_list = [self.plot_tsne(2).data,
                    self.plot_coherence().data,
                    self.plot_likelihood(30).data]

        for i in fig_list[0]:
            fig.add_trace(i, 1, 1)        
    
        fig.add_trace(fig_list[1][0], 2, 1)
        fig.add_trace(fig_list[1][1], 3, 1)


        for i in fig_list[2]:
            fig.add_trace(i, 2, 2)

        fig.update_layout(width =w, height =h )
    
        return fig


# LDA Topic Modeling by decade
class LDAPipeline():
    
    def __init__(self,
                 prep,
                 gs = False,
                 decades = [
        1960, 1970, 1980, 1990, 2000, 2010, 2020
    ]):
        self.models = {
            decade : LDATopicModeling(
                decade = decade,
                lang_preprocess = prep,
                epochs = 10,
                grid_search = gs) for decade in decades}
        
    def get_metrics(self):
        # compute metrics
        metrics = {
                 'decade': [],
                 'coherence': [],
                 'perplexity': []
        }
        for decade, model in self.models.items():
            metrics['decade'].append(decade)
            metrics['coherence'].append(model.get_coherence())
            metrics['perplexity'].append(model.get_perplexity())
        # create the dataframe
        df_m = pd.DataFrame(metrics)
        df_m.set_index('decade')
        return df_m
        
        
    def lda_info(self, decade):
        lda_model = self.models[decade]

        print("Perplexity: ", lda_model.get_perplexity())
        print("Coherence: ", lda_model.get_coherence())
        lda_model.plot_tsne()
        return lda_model.dashboard_LDAvis()