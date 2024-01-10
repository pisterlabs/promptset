#!/usr/bin/env python
# coding: utf-8

import gensim
import pickle
import argparse
import logging
import time
import yaml
from utils import *
from gensim.models import LdaModel,TfidfModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from dataset import DocDataset
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser('LDA topic model')
parser.add_argument('--config',type=str,default=None,help='Config YAML file containing data paths and hyperparameters')
parser.add_argument('--taskname',type=str,default='cnews10k',help='Taskname e.g cnews10k')
parser.add_argument('--no_below',type=int,default=5,help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above',type=float,default=0.3,help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--num_iters',type=int,default=1000,help='Number of iterations (set to 100 as default, but 1000+ is recommended.)')
parser.add_argument('--n_topic',type=int,default=5,help='Num of topics')
parser.add_argument('--bkpt_continue',type=bool,default=False,help='Whether to load a trained model as initialization and continue training.')
parser.add_argument('--ngrams',type=bool,default=True,help='Whether to include bigrams and trigrams in dictionary')
parser.add_argument('--use_tfidf',type=bool,default=False,help='Whether to use the tfidf feature for the BOW input')
parser.add_argument('--rebuild',type=bool,default=False,help='Whether to rebuild the corpus, such as tokenization, build dict etc.(default True)')
parser.add_argument('--auto_adj',action='store_true',help='To adjust the no_above ratio automatically (default:rm top 20)')

args = parser.parse_args()

def main():
    global args
    
    config = args.config
    taskname = args.taskname
    no_below = args.no_below
    no_above = args.no_above
    num_iters = args.num_iters
    n_topic = args.n_topic
    n_cpu = cpu_count()-2 if cpu_count()>2 else 2
    bkpt_continue = args.bkpt_continue
    ngrams = args.ngrams
    use_tfidf = args.use_tfidf
    rebuild = args.rebuild
    auto_adj = args.auto_adj

    txtpath = None
    stopwords = None
    if config:
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
            taskname = config['taskname']
            txtpath = config['txtpath']
            stopwords = config['stopwords']
            n_topic = config['n_topic'] if 'n_topic' in config else n_topic
            lower = config['lower'] if 'lower' in config else None
            upper = config['upper'] if 'upper' in config else None
            n_trials = config['n_trials'] if 'n_trials' in config else 1

    docSet = DocDataset(taskname,txtpath=txtpath,stopwords=stopwords,no_below=no_below,no_above=no_above,rebuild=rebuild,ngrams=ngrams)
    if auto_adj:
        no_above = docSet.topk_dfs(topk=20)
        docSet = DocDataset(taskname,txtpath=txtpath,stopwords=stopwords,no_below=no_below,no_above=no_above,rebuild=rebuild,ngrams=ngrams,use_tfidf=False)
    
    taskname = f'lda_{taskname}'
    model_name = 'LDA'
    msg = 'bow' if not use_tfidf else 'tfidf'

    # Run experiment to find optimal no. of topics
    if not lower or not upper:
        lower, upper = n_topic, n_topic
    for i in range(n_trials):
        tp, cv, td = [], [], []
        for n_topic in range(lower,upper+1):
            run_name= '{}_K{}_{}_{}'.format(model_name,n_topic,taskname,msg)
            makedir('logs')
            makedir('ckpt')
            loghandler = [logging.FileHandler(filename=f'logs/{run_name}.log',encoding="utf-8")]
            logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(message)s',handlers=loghandler)
            logger = logging.getLogger(__name__)


            if bkpt_continue:
                print('loading model ckpt ...')
                lda_model = gensim.models.ldamodel.LdaModel.load('ckpt/{}.model'.format(run_name))


            # Training
            print('Start Training ...')

            if use_tfidf:
                tfidf = TfidfModel(docSet.bows)
                corpus_tfidf = tfidf[docSet.bows]
                corpus = list(corpus_tfidf)
            else:
                corpus = list(docSet.bows)
            lda_model = LdaMulticore(corpus,num_topics=n_topic,id2word=docSet.dictionary,alpha='asymmetric',passes=num_iters,workers=n_cpu,minimum_probability=0.0)
            #lda_model = LdaModel(corpus,num_topics=n_topic,id2word=docSet.dictionary,alpha='asymmetric',passes=num_iters)
            save_name = f'./ckpt/LDA_{taskname}_tp{n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}.ckpt'
            #lda_model.save(save_name)


            # Evaluation
            print('Evaluation ...')
            topics_dict = get_topic_words(ldamodel=lda_model,n_topic=n_topic,topn=15,vocab=docSet.dictionary, showWght=True)
            topic_words = [[x[0] for x in topic] for topic in topics_dict.values()]

            (cv_score, w2v_score, c_uci_score, c_npmi_score),_ = calc_topic_coherence(topic_words,docs=docSet.docs,dictionary=docSet.dictionary,taskname=taskname)

            topic_diversity = calc_topic_diversity(topic_words)

            result_dict = {'cv':cv_score,'w2v':w2v_score,'c_uci':c_uci_score,'c_npmi':c_npmi_score}
            logger.info('Topics:')

            save_directory = f'results/{taskname}'
            makedir(save_directory)
            with open(f'{save_directory}/{taskname}_tp{n_topic}.txt', 'w') as f:
                for idx,words in enumerate(topic_words):
                    logger.info(f'##{idx:>3d}:{words}')
                    print(f'##{idx:>3d}:{words}')
                    f.write(f'##{idx:>3d}:{words}\n')

                for measure,score in result_dict.items():
                    logger.info(f'{measure} score: {score}')
                    print(f'{measure} score: {score}')
                    f.write(f'{measure} score: {score}\n')

                logger.info(f'topic diversity: {topic_diversity}')
                print(f'topic diversity: {topic_diversity}')
                f.write(f'topic diversity: {topic_diversity}\n')

                topic_dist = get_topics_in_corpus(lda_model, corpus)
                topics, topic_dist = show_topic_dist_in_corpus(topic_dist, topk=n_topic)
                print(f'Topic distribution:\n{topic_dist}')
                f.write(f'Topic distribution:\n{topic_dist}')
                f.close()

                if lower == upper:
                    # Create word clouds
                    create_wordclouds(topics, topics_dict, save_directory)

            cv.append(cv_score)
            td.append(topic_diversity)
            tp.append(n_topic)
        if lower != upper:
            with open(f'{save_directory}/lda_cv_{i}.pkl', 'wb') as f:
                pickle.dump(cv, f)
            with open(f'{save_directory}/lda_td_{i}.pkl', 'wb') as f:
                pickle.dump(td, f)
            with open(f'{save_directory}/lda_tp_{i}.pkl', 'wb') as f:
                pickle.dump(tp, f)
            print(f'Pickle dumped at trial {i}, #topics {n_topic}')

def get_topic_words(ldamodel,topn=15,n_topic=10,vocab=None,fix_topic=None,showWght=False):
    topics = {}
    def show_one_tp(tp_idx):
        if showWght:
            return [(vocab.id2token[t[0]],t[1]) for t in ldamodel.get_topic_terms(tp_idx,topn=topn)]
        else:
            return [vocab.id2token[t[0]] for t in ldamodel.get_topic_terms(tp_idx,topn=topn)]
    if fix_topic is None:
        for i in range(n_topic):
            topics[i] = show_one_tp(i)
    else:
        topics[fix_topic] = show_one_tp(fix_topic)
    return topics

def get_topics_in_corpus(ldamodel, corpus):
    return [get_dominant_topic(row) for row in ldamodel[corpus]]

def get_dominant_topic(row):
    return np.array(row)[:,1].argmax()

if __name__ == '__main__':
    main()
