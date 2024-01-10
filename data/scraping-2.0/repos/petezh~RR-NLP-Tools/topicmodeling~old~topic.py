
# processing
from operator import methodcaller
import csv
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

blacklist = next(csv.reader(open("blacklist.csv", 'r')))

def main():
    
    execute("input.csv", "out.csv", 10, 70)
    

def execute(inPath, outPath, wordBound, charBound):

    inFile = open(inPath, 'r')
    inReader = csv.reader(inFile)

    outFile = open(outPath, 'w')
    outWriter = csv.writer(outFile)

    
    docTokens = dict()


    next(inReader)
    for inRow in inReader:

        charDist = int(inRow[0])
        wordDist = int(inRow[1])

        if wordDist < wordBound and charDist < charBound:
        
            #predTerm, subTerm, objTerm = map(methodcaller("split", ":"), inRow[2:5])
            #allTerms = predTerm + subTerm + objTerm

            subTerm, objTerm = map(methodcaller("split", ":"), inRow[3:5])
            allTerms = subTerm + objTerm
            
            tokens = list()
            
            for term in allTerms:
                if term.isalpha() and not term in blacklist and len(term) > 2:
                    tokens.append(term)
            
            docID = inRow[5]
            
            if docID in docTokens:
                docTokens[docID] += tokens
            else:
                docTokens[docID] = tokens

    data = list(docTokens.values())


    # get bigrams
    bigram = gensim.models.Phrases(data, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    data = [bigram_mod[doc] for doc in data]

    id2word = corpora.Dictionary(data)
    texts = data

    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    vis
    
if __name__ == "__main__":
    main()


