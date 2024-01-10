# processing
import operator
from operator import methodcaller
import csv
import re
import numpy as np
import pandas as pd
from pprint import pprint
import string
import math
import sqlite3

# gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import HdpModel

# plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

#sci-kit
from sklearn import feature_extraction

import warnings
warnings.filterwarnings(action='once')


resultFile = open("results.csv",'a')
results = csv.writer(resultFile,lineterminator ='\n')




def main():
    i = 1
    numtops = [5, 10, 20]
    ratios = [0.01, 0.02]
    resistances = [0.8]
    for nt in numtops:
        for rat in ratios:
            for res in resistances:
                print("Test #" + str(i) + " with numtops " + str(nt) + " with ratio " + str(rat) + " and res " + str(res))
                
                test(rat, res, nt)
                i += 1

    resultFile.close()
    
def test(rat, res, nt):

    global origCorpus

    corpus = origCorpus.copy()
    
    id_topic_ratio = rat
    resistance = res
    done = False
    numTops = nt
    
    topicPath = "data\\topics_init" + str(nt) + "_rat"+str(rat)+"_res_"+str(res)+".csv"
    relationPath = "data\\relations_init" + str(nt) + "_rat"+str(rat)+"_res_"+str(res)+".csv"

    topicFile = open(topicPath, 'w')
    topicOut =  csv.writer(topicFile, lineterminator = '\n')
    topicOut.writerow(["", "run", "topic", "terms", "p"])


    relationFile = open(relationPath, 'w')
    relationOut = csv.writer(relationFile, lineterminator = '\n')
    relationOut.writerow(["run", "topic", "no IDs", "ID/strength"])

    run = 1
    totalTopics = 0
    averageCoherence = 0
    badIDs = docIDs

    while not done:
        
        print("Run #" + str(run))
        
        doc2topic = dict()
        topic2doc = dict()
        
        
        oldIDs = badIDs.copy()
        badIDs = list()
        
        totalTopics += numTops
        
        #perform LDA
        hdp = HdpModel(corpus, dictionary, T=numTops)

        lda_model = hdp.suggested_lda_model()
        
        coherenceModel = CoherenceModel(model=lda_model, texts=data, dictionary=dictionary, coherence='c_v')
        coherence = coherenceModel.get_coherence()
        averageCoherence = ((totalTopics-numTops) * averageCoherence + numTops*coherence)/totalTopics
        
        # tag documents
        for ID in oldIDs:
            
            doc = docTokens[ID]
            vec = dictionary.doc2bow(doc)

            store = lda_model[vec]

            bestRel = 0

            # build relations
            for pair in store:
                
                bestRel = max(bestRel, pair[1])

                if pair[0] in topic2doc:
                    topic2doc[pair[0]] += [(ID, pair[1])]
                else:
                    topic2doc[pair[0]] = [(ID, pair[1])]

            # collect bad docs    
            if bestRel < resistance:

                badIDs.append(ID)
        
        
        #write terms
        
        top_words_per_topic = []
        for t in range(lda_model.num_topics):
            top_words_per_topic.extend([(run, t, ) + x for x in lda_model.show_topic(t, topn = 10)])

            
        terms = pd.DataFrame(top_words_per_topic, columns=['Run', 'Topic', 'Word', 'P']).to_csv(topicPath, mode='a', header=False)
        
        
        # print relations
        for topic in topic2doc:
            relationOut.writerow([run, topic, len(topic2doc[topic])]+ sorted(topic2doc[topic], key=operator.itemgetter(1), reverse=True))
        
        
        
        # done?
        if len(badIDs) == 0:
            done = True
            print("Done!")
        
        # if not, build new corpus
        else:
            print("Remaining: " + str(len(badIDs)))
            corpus = [dictionary.doc2bow(docTokens[docID]) for docID in badIDs]
            len(corpus)
            numTops = math.ceil(len(badIDs) * id_topic_ratio)
            run += 1

    results.writerow([nt, rat, res, averageCoherence, totalTopics])
        
    topicFile.close()
    relationFile.close()






# preprocessing
blFile= open("tools\\blacklist.csv", 'r')
blacklist = [t.strip() for t in next(csv.reader(blFile))]
blFile.close()
levels = [1, 2, 3]

# format [term, orig, sentence, docID]
inPath = "raw.csv"

inFile = open(inPath, 'r')
inReader = csv.reader(inFile)

docTokens = dict()

# ignore headers
next(inReader)

for inRow in inReader:
    
    term = inRow[0]
    sentence = inRow[2]
    docID = inRow[3]
    
    # find acceptable tokens only
    token = "_".join([t for t in term.split(":") if re.match(r'[^\W\d]*$', t) and not t in blacklist])
    
    # calculate new term level
    level = token.count("_")
    
    # if acceptable, add to dictionary
    if level in levels and not token in blacklist and len(token) > 0:
        if docID in docTokens:
            docTokens[docID] += [token]
        else:
            docTokens[docID] = [token]

docIDs = list(docTokens.keys())
data = list(docTokens.values())

dictionary = corpora.Dictionary(data)
texts = data

origCorpus = [dictionary.doc2bow(text) for text in texts]




if __name__ == "__main__":
    main()
    
