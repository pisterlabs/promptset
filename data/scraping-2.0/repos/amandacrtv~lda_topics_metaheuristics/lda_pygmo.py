import numpy as np
import os
from gensim import corpora, models
import multiprocessing
from ctypes import *
from gensim.models.coherencemodel import CoherenceModel
import pygmo as pg
import pandas as pd
import traceback

from datetime import datetime

language = 'python'
year = '2019'

min_topic = 10
max_topic = 30

iters = 100
passes = 5

filter_extremes = 10
# filter_extremes = 5

type_evo = 'PSO'
# columns = ['Gen', 'Fevals', 'Best', 'dx', 'df']
# columns = ['Gen', 'Fevals', 'Best', 'Improvement']
columns = ['Gen', 'Fevals', 'gbest', 'Mean Vel.', 'Mean lbest', 'Avg. Dist.']
# columns = ['Gen', 'Fevals', 'gbest', 'Mean Vel.', 'Mean lbest', 'Avg. Dist.']
# columns = ['Fevals', 'Best', 'Current', 'Mean range', 'Temperature']
# columns = ['Gen', 'Fevals', 'Best', 'F', 'CR', 'dx', 'df']


foldermain = f'./Experiments_more_5stars/{language}/{year}/'
dataset = f'./yearly_dataset/5_stars/en_cleaned_all_more_{language}_{year}.csv'

foldermodels = foldermain + f'/models/'
folderreprops = foldermain + f'/repoprops/'

seed = 0

# save_csv_log = f'./log/{type_evo}_log.csv'
save_csv_log = f'./log/{type_evo}_more_5stars_log_{language}_{year}.csv'

class lda_model:

    def calc_fitness(self, NTopic):
        numoftopics = int(NTopic[0])
        topic_num.value = numoftopics
        al = float(round(NTopic[1], 3))
        bet = float(round(NTopic[2], 3))

        log = str(count.value)+' '+str(numoftopics)  + ' ' + str(al) + ' ' + str(bet) + "\n"
        print(log)
        logfile.write(log)

        model = models.ldamodel.LdaModel(corpus2, num_topics=numoftopics, id2word=dictionary,
                                                        iterations=iters, passes=passes, chunksize=5000, alpha=al, eta=bet, random_state=seed)

        print("Created Model::" + str(count.value))

        coherence_model_lda = CoherenceModel(model=model, texts=docs, dictionary=dictionary, coherence="c_v")
        coherence_lda = coherence_model_lda.get_coherence()

        log = "SCORE::" + str(coherence_lda) +'\n'
        print(log)
        logfile.write(log)
        count.value += 1
        print("COUNT::" + str(count.value)+'\n')
        model.clear()
        return coherence_lda

    def get_bounds(self):
        return ([min_topic, 0.001, 0.001], [max_topic, 0.5, 0.5])
    
    def fitness(self, NTopic):
        numoftopics = int(NTopic[0])
        al = float(round(NTopic[1], 3))
        bet = float(round(NTopic[2], 3))
        log = str(numoftopics) + ' ' + str(al) + ' ' + str(bet) + "\n"
        if not log in coScore:
            coScore[log] = self.calc_fitness(NTopic) * -1
        else:
            print(log)
        return [coScore[log]]

def log_to_csv(log):
    df = pd.DataFrame(log, columns=columns)
    if type_evo == 'PSO' or type_evo == 'GPSO':
        df['gbest'] = [val * -1 for val in df['gbest']]
    else:
        df['Best'] = [val * -1 for val in df['Best']]
    df.to_csv(save_csv_log, index=False)

def format_individual(NTopic):
    return [int(NTopic[0]), float(round(NTopic[1], 3)),  float(round(NTopic[2], 3))] 

def eval_func_JustModel(NTopic):
    numoftopics = int(NTopic[0])
    topic_num.value = numoftopics
    al = float(round(NTopic[1], 3))
    bet = float(round(NTopic[2], 3))
    log = str(numoftopics) + ' '+ str(al) + ' ' + str(bet) + "\n"
    print(log)
    logfile.write(log)
    model = models.ldamodel.LdaModel(corpus2, num_topics=numoftopics, id2word=dictionary, chunksize=5000, passes=int(iters/10), iterations=iters, alpha=al,eta=bet, random_state=seed)
    model.save(foldermodels+str(numoftopics) +'_'+str(iters) + '.model')
    coherence_model_lda = CoherenceModel(model=model, texts=docs, dictionary=dictionary, coherence="c_v")
    coherence_lda = coherence_model_lda.get_coherence()
    log = "\nFINAL SCORE (10 PASSES) :: " + str(coherence_lda) +'\n'
    print(log)
    logfile.write(log)

def apply_and_save_model():
    try:
        dictionary = corpora.Dictionary.load(foldermodels + 'MultiCore.dict')
        modelfile = ''
        for f in os.listdir(foldermodels):
            if ('.model' in f):
                filetype = f.split('.')[-1]
                if filetype not in ['state','id2word','npy']:
                    modelfile = f

        print(modelfile)
        model_test = models.ldamodel.LdaModel.load(foldermodels + modelfile)

        # show word probabilities for each topic
        X = model_test.show_topics(num_topics=max_topic, num_words=5, log=False, formatted=False)
        test = [(x[0], [y[0] for y in x[1]]) for x in X]
        topicDesc = {}

        for t in test:
            topicstr = ' + '.join(str(e) for e in t[1])
            topicDesc[t[0]] = topicstr

        for k in topicDesc.keys():
            line = str(k) + ',' + topicDesc[k]
            fout = open(folderreprops + 'topickeys.csv', 'a', encoding="utf8")
            fout.write(line + '\n')
            fout.close()

        df = pd.read_csv(dataset, usecols=['full_name', 'filtered'])    
        docdict = {}

        for _, row in df.iterrows():
            docdict[row['full_name']] = row['filtered'].split()

        docTopicDict = {}

        for d in docdict.keys():
            docProbs = model_test[[dictionary.doc2bow(docdict[d])]]
            currrdocProb = [0] * max_topic
            for p in docProbs[0]:
                currrdocProb[p[0]] = p[1]
                doc_topic = np.array(currrdocProb)
                topic = np.array(doc_topic).argmax()
                docTopicDict[d] = topic

        for k in docTopicDict:
            line = k + ',' + str(docTopicDict[k])
            fout = open(folderreprops + 'doctopic.csv', 'a', encoding="utf8")
            fout.write(line + '\n')
            fout.close()

        fnames = docTopicDict.keys()
        comb = open(folderreprops + 'comb2desc.csv', 'a', encoding="utf8")

        for doc in docdict.keys():
            if (doc in fnames):
                currtopic = docTopicDict[doc]
                newline = doc + ',' + str(currtopic) + ',' + str(topicDesc[currtopic])
            else:
                newline = doc + ',TOPICASSIGNED,TOPICWORDS'
            comb.write(newline + '\n')
        comb.close()

    except Exception as e:
        print("Error Occured while combining topics with documents")
        traceback.print_exc()
        quit()

if __name__ == '__main__':

    manager = multiprocessing.Manager()
    df = pd.read_csv(dataset, usecols=['filtered'])

    # threshold = 10000
    # df = df.sample(threshold, random_state=seed)

    corpus = manager.list()
    count = manager.Value(c_int, 0)
    docs = manager.list()
    rc = 0

    topic_num = manager.Value(c_int, 0)
    corpus2 = manager.list()
    coScore = manager.dict()

    print(f'Document {dataset} shape {df.shape}')

    corpus = df['filtered'].tolist()
    docs = [doc.split() for doc in corpus]

    dictionary = corpora.Dictionary(docs)

    dictionary.filter_extremes(no_below=filter_extremes, no_above=0.8)
    
    print(f'Language: {language}')
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    corpus2 = [dictionary.doc2bow(doc) for doc in docs]
    dictionary.save(foldermodels+'MultiCore.dict')
    corpora.MmCorpus.serialize(foldermodels+'MultiCoreCorpus.mm', corpus2)

    logfile = open(foldermodels+'/log.txt','w', encoding='utf-8')

    prob = pg.problem(lda_model())
    
    # algo = pg.algorithm(pg.de(gen = 15, seed = seed, variant=7, F=0.4, CR=0.9)) #300 Feval
    # algo = pg.algorithm(pg.sga(gen = 15, seed = seed))
    # algo = pg.algorithm(pg.pso_gen(gen = 15, seed = seed, variant=1))
    # algo = pg.algorithm(pg.pso(gen = 15, seed = seed, variant=1))
    algo = pg.algorithm(pg.pso(gen = 5, seed = seed, variant=1))
    # algo = pg.algorithm(pg.simulated_annealing(seed = seed, n_T_adj=5, n_range_adj=4, bin_size=5))
    # algo = pg.algorithm(pg.sade(gen = 15, seed = seed, variant=7, variant_adptv=2))

    algo.set_verbosity(1)

    start_time = datetime.now()

    print(f'LDA optmization using {algo.get_name()} started at', start_time)

    print("Starting Mutations::")
    print("COUNT::" + str(count.value))

    pop = pg.population(prob = prob, size = 20, seed = seed)
    # pop = pg.population(prob = prob, size = 1, seed = seed)

    pop = algo.evolve(pop) 

    end_time = datetime.now()

    print(f'LDA optmization using {algo.get_name()} finished at', end_time)

    print(f'LDA optmization using {algo.get_name()} done in ', end_time - start_time)
    print(f'Dataset {dataset}')
    print(f'min_topic: {min_topic} max_topic: {max_topic}')

    print(pop.champion_f * -1)
    print(format_individual(pop.champion_x))

    # log = algo.extract(pg.de).get_log()
    # log = algo.extract(pg.sga).get_log()
    # log = algo.extract(pg.pso_gen).get_log()
    log = algo.extract(pg.pso).get_log()
    # log = algo.extract(pg.simulated_annealing).get_log()
    # log = algo.extract(pg.sade).get_log()

    log_to_csv(log)

    print(f'Number of Models generated: {count.value}')
    fo = open(foldermodels+"bestindividual", "a", encoding="utf8")
    eval_func_JustModel(pop.champion_x)
    line = f'Best individual coherence: {pop.champion_f * -1} individual: {format_individual(pop.champion_x)}'
    fo.write(line)
    logfile.write(line)
    logfile.write(f'\nTotal time for LDA optmization using {algo.get_name()}: {end_time - start_time}')
    logfile.write(f'\nDataset {dataset}')
    logfile.write(f'\nmin_topic: {min_topic} max_topic: {max_topic}')
    fo.close()
    logfile.close()

    apply_and_save_model()

    