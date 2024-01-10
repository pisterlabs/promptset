
# coding: utf-8

'''
This class implements necessary helper methods for performing LDA, Agglomerative Clustering and KMeans clustering of features. Most of the methods are self-explanatory. Comments are available wherever necessary.
'''

#import statements
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt                   # For graphics
import seaborn as sns
import numpy as np
import pylab as pl
#import plotly.plotly as py #For World Map
#import plotly.graph_objs as go
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#init_notebook_mode(connected=True)

from sklearn.preprocessing import StandardScaler  # For scaling dataset
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation #For clustering
from sklearn.mixture import GaussianMixture #For GMM clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

import pylab as pl
from collections import OrderedDict
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

import gensim
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim

class Clustering:
    def __init__(self,lda_params,hierarchical_params,results_path,content):
        self.lda_params = lda_params
        self.hierarchical_params = hierarchical_params
        self.hierarchical_num = None
        self.lda_num = None
        self.perplexity_lda = None
        self.coherence_lda = None
        self.cophenet = None
        self.kmeans_num = None
        self.path = results_path
        self.content = content
        
    def compute_coherence_models(self,Lda,doc_term_matrix,dictionary,start,stop,step,notebook_corpus):
        models = []
        coherence_scores = []
        for num in range(start,stop,step):        
            print("num of topics: ",num)

            ldamodel = Lda(doc_term_matrix, num_topics=num, id2word = dictionary, passes=50,random_state=0)
            models.append(ldamodel)
            # Compute Coherence Score
            coherence_model_lda = CoherenceModel(model=ldamodel, texts=notebook_corpus, dictionary=dictionary, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            coherence_scores.append(coherence_lda)

        return models,coherence_scores
    
    def get_best_lda_results(self,top_notebook_split,start,stop,step,top_words):
        # Term Document Frequency
        # Creating the term dictionary of our corpus, where every unique term is assigned an index.  
        dictionary = corpora.Dictionary(top_notebook_split)

        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above. 
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in top_notebook_split]

        # Human readable format of corpus (term-frequency)
        #[[(dictionary[id], freq) for id, freq in cp] for cp in doc_term_matrix[:1]]

        # Creating the object for LDA model using gensim library
        Lda = gensim.models.ldamodel.LdaModel

        models,coherence_scores = self.compute_coherence_models(Lda,doc_term_matrix,dictionary,start,stop,step,top_notebook_split)
        print("model, coherence_score: ")
        for i in range(len(models)):
            print(models[i],coherence_scores[i])
        best_model = models[coherence_scores.index(max(coherence_scores))]
        lda_num = best_model.num_topics
        print("Best number of topics: ",lda_num)

        # Running and Training LDA model on the document term matrix
        #************* what is the idea number of topics is the question! *************
        print("Running the best model with %d..." %lda_num)
        ldamodel = Lda(doc_term_matrix, num_topics=lda_num, id2word = dictionary, passes=50,random_state=0)

        # Results 
        print("Top 10 words per topic...")
        for each in ldamodel.print_topics(num_words=top_words):
            #print_topics is alias for show_topics method
            #print(each)  #print vocabulary with their weights
            print("vocabulary: ",[e.split('*')[1].replace('"','').strip() for e in each[1].split('+')])

        '''
        for each in ldamodel.show_topics():
            topn=len(dictionary) 3word distribution
            print(each)
        '''

        # Compute Perplexity
        perplexity_lda = ldamodel.log_perplexity(doc_term_matrix)
        print('Perplexity: ',perplexity_lda)  # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=top_notebook_split, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('Coherence Score: ', coherence_lda)
        
        #pyLDAvis.enable_notebook()
        #vis = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary)
        #vis
        
        return (lda_num,perplexity_lda,coherence_lda)
    
    def get_best_hierarchical_results(self,vect_corpus,full_dendrogram,truncated_dendrogram,max_d,linkage_metric,affinity_metric):
        # generate the linkage matrix
        doc_matrix = np.nan_to_num(vect_corpus)
        Z = linkage(vect_corpus, linkage_metric) #ward variance minimization algorithm

        #explain cophenet distance
        c, coph_dists = cophenet(Z, pdist(vect_corpus))
        if full_dendrogram:
            self.plot_full_dendrogram(Z)
        if truncated_dendrogram:
            self.plot_truncated_dendrogram(Z,max_d)

        last = Z[-10:, 2]
        last_rev = last[::-1] #last convergence distances
        idxs = np.arange(1, len(last) + 1)
        sns.set(style="white")
        sns.lineplot(idxs, last_rev)
        acceleration = np.diff(last, 2)  # 2nd derivative of the distances
        acceleration_rev = acceleration[::-1]
        sns.lineplot(idxs[:-2] + 1, acceleration_rev)
        plt.savefig(self.path+'hierarchical_'+self.content+'.png',bbox_inches="tight")
        plt.show()
        n_clusters = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
        print("Running the best hierarchical clustering model with predicted cluster number... %d" %n_clusters)
        self.agglomerative_clustering(n_clusters,affinity_metric,linkage_metric,vect_corpus)
        return (n_clusters,c)
    
    def KMeans_model(self,clust,number_of_clusters,vect_corpus,vect_feature_names):
        self.kmeans_num = int(np.mean([self.lda_num,self.hierarchical_num]))
        n_clusters = self.kmeans_num
        if clust == 'default':
            n_clusters = self.kmeans_num
        else:
            n_clusters = number_of_clusters
        print("n_clusters for kmeans ",n_clusters)
        kmeans = KMeans(n_clusters=n_clusters).fit(vect_corpus)
        y = kmeans.labels_
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        cluster_words = []
        for i in range(n_clusters):
            top_words = []
            for ind in order_centroids[i, :8]:
                top_words.append(vect_feature_names[ind])
            cluster_words.append(top_words)
        self.plot_kmeans_using_pca(vect_corpus,y,n_clusters,cluster_words)
        return (cluster_words,y)
    
    def agglomerative_clustering(self,hierarchical_num,affinity_metric,linkage_metric,vect_corpus):
        cluster = AgglomerativeClustering(n_clusters=hierarchical_num, affinity=affinity_metric, linkage=linkage_metric)  
        y = cluster.fit(vect_corpus)   #y or cluster.labels_
        self.plot_using_pca(vect_corpus,y.labels_,hierarchical_num)
        
    def plot_full_dendrogram(self,Z):    
        # calculate full dendrogram
        plt.figure(figsize=(25, 10))
        sns.set(style="whitegrid")
        plt.grid(None)
        #plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('cluster size')
        plt.ylabel('distance')
        dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=3.,  # font size for the x axis labels
        )
        plt.savefig(self.path+'hierarchical_full_dendrogram_'+self.content+'.png',bbox_inches="tight")
        plt.show()
        
    def plot_truncated_dendrogram(self,Z,max_d):
        #plt.title('Hierarchical Clustering Dendrogram (truncated)')
        sns.set(style="whitegrid")
        plt.grid(None)
        plt.xlabel('cluster size')
        plt.ylabel('distance')
        dendrogram(
            Z,
            truncate_mode='lastp',  # show only the last p merged clusters
            p=12,  # show only the last p merged clusters
            show_leaf_counts=False,  # otherwise numbers in brackets are counts
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,  # to get a distribution impression in truncated branches

        )

        plt.axhline(y=max_d, c='k')
        plt.savefig(self.path+'hierarchical_truncated_dendrogram_'+self.content+'.png',bbox_inches="tight")
        plt.show()

    def plot_using_pca(self,vect_corpus,y,n_clusters):
        pca = PCA(n_components=3,random_state=0).fit(vect_corpus)
        pca_2d = pca.transform(vect_corpus)
        
        markers = ['+','o','*','^','D','X','x','p','s','<','>','h','H','d','.','8','4','3','2','1']
        colors = ['r','g','b','m','c','k','y','orange','lime','navy','yellow','olive','hotpink']
        cluster_colors = np.arange(n_clusters)
        sns.set(style="white")
        fig, ax = plt.subplots(1,1,figsize=(12,8),subplot_kw={'projection': '3d'})
        plt.rcParams["axes.linewidth"]  = 1.25
        plt.rcParams["axes.grid"] = True
        for i in range(0, pca_2d.shape[0]):
            cluster_id = y[i]
            ax.scatter(pca_2d[i,0], pca_2d[i,1],pca_2d[i,2], c=colors[cluster_id],marker=markers[cluster_id],s=12)

        #handles, labels = plt.gca().get_legend_handles_labels()
        #by_label = OrderedDict(zip(labels, handles))
        #pl.legend(by_label.values(), by_label.keys())
        #pl.title('Notebook with '+str(clusters) +' clusters and known outcomes')
        plt.tight_layout()
        ax.view_init(azim=-5)
        plt.savefig(self.path+'hierarchical_pca_view_'+self.content+'.png',bbox_inches="tight")
        plt.show()
        plt.gcf().clear()
        
    def plot_kmeans_using_pca(self,vect_corpus,y,n_clusters,cluster_words):
        pca = PCA(n_components=3,random_state = 0).fit(vect_corpus)
        pca_2d = pca.transform(vect_corpus)
        #print(pca_2d)
        markers = ['+','o','*','^','D','X','x','p','s','<','>','h','H','d','.','8','4','3','2','1']
        colors = ['c','m','b','g','r','k','y','orange','lime','navy','yellow','olive','hotpink']
        cluster_colors = np.arange(n_clusters)
        fig, ax = plt.subplots(1,1,figsize=(12,8),subplot_kw={'projection': '3d'})
        sns.set(style="white")
        print("plotting pca for kmeans..")
        plt.rcParams["axes.linewidth"]  = 1.25
        plt.rcParams["axes.grid"] = True
        for i in range(0, pca_2d.shape[0]):
            cluster_id = y[i]
            #c=cluster.labels_ ,label=cluster_words[cluster_id]
            ax.scatter(pca_2d[i,0], pca_2d[i,1], pca_2d[i,2], c=colors[cluster_id],marker=markers[cluster_id],label=cluster_words[cluster_id],s=12)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        #plt.title('Notebook with '+str(n_clusters) +' clusters and known outcomes')
        plt.tight_layout()
        #ax.view_init(azim=-5)
        plt.savefig(self.path+'kmeans_pca_'+self.content+'.png',bbox_inches="tight")
        plt.show()
        plt.gcf().clear()
        
    def get_best_clusters_prediction(self):
    
        print("Running lda...")
        #lda clustering parameters
        start = self.lda_params[0]
        stop = self.lda_params[1]
        step = self.lda_params[2]
        top_words = self.lda_params[3] #top words to return per vocabulary
        corpus = self.lda_params[4]
        (lda_num,perplexity_lda,coherence_lda) = self.get_best_lda_results(corpus,start,stop,step,top_words)
        print("Best number of clusters using lda model: ", lda_num)
        print("perplexity metric: ",perplexity_lda)
        print("coherence metric: ",coherence_lda)
        self.lda_num = lda_num
        self.coherence_lda = coherence_lda
        self.perplexity_lda = perplexity_lda
        
        print("Running hierarchical...")
        #hierarchical clustering parameters
        max_d = self.hierarchical_params[0] #maximum distance to determine our clusters for truncated plot
        full_dendrogram = self.hierarchical_params[1] #for plotting
        truncated_dendrogram = self.hierarchical_params[2] #for plotting
        linkage_metric = self.hierarchical_params[3]
        affinity_metric = self.hierarchical_params[4]
        vect_corpus =  self.hierarchical_params[5]
        (hierarchical_num,cophenet) = self.get_best_hierarchical_results(vect_corpus,full_dendrogram,truncated_dendrogram,max_d,
                                                                         linkage_metric,affinity_metric)
        print("Best number of clusters using hierarchical clusters: ", hierarchical_num)
        print("cophenet metric: ",cophenet)
        self.hierarchical_num = hierarchical_num
        self.cophenet = cophenet
        
        return (self.lda_num,self.hierarchical_num)
    
    

