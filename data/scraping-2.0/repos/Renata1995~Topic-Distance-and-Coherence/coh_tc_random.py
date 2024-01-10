import random
import sys

import numpy
from gensim import corpora
from utils.corpus import arg_tfidf

from coherence.umass import TopicCoherence

# python random_tc.py <dname> <word_count> <sample_times>
# <word_count>: the number of words that need to be randomly generated
# <sample_times>: the repetition times of the topic coherence calculation

if len(sys.argv) <= 1:
    dname = "reuters_LDA"
else:
    dname = sys.argv[1]

if len(sys.argv) <= 2:
    word_count = 10
else:
    word_count = int(sys.argv[2])

if len(sys.argv) <= 3:
    sample_times = 5
else:
    sample_times = int(sys.argv[3])

if len(sys.argv) <= 4:
    tfidf = False
else:
    if sys.argv[4] == "t":
        tfidf = True
    else:
        tfidf = False

dictionary = corpora.Dictionary.load(dname + "/dict.dict")
print "Load dictionary",
print dictionary

# Load corpus
if tfidf:
    corpus_fname = dname + "/bow_corpus.mm"
    print "Load Corpus File " + corpus_fname
    bow_corpus = corpora.MmCorpus(corpus_fname)
    corpus = arg_tfidf(bow_corpus, dictionary)
else:
    corpus_fname = dname + '/binary_corpus.mm'
    print "Load Corpus File " + corpus_fname
    corpus = corpora.MmCorpus(corpus_fname)

# transfer each doc in the corpus into a dictionary
corpus_dict =[]
for doc in corpus:
    corpus_dict.append(dict(doc))
dictlen = len(dictionary)

tc = TopicCoherence()

tc_results = []
words_list = []
if tfidf:
    ofile = open(dname+"/tc_rand_tfidf_"+str(word_count)+".txt", "w")
    epsilon = 0.0001
else:
    ofile = open(dname+"/tc_rand_"+str(word_count)+".txt", "w")
    epsilon = 1

for i in range(sample_times):
    random_words = []
    # generate random numbers
    for n in range(word_count):
        word = random.randint(1, dictlen-1)
        while word in random_words:
            word = random.randint(0, dictlen-1)
        random_words.append(word)

    keylist = []
    for key in random_words:
        keylist.append(dictionary[key])
    words_list.append(keylist)

    # calculate topic coherence based on randomly generated words
    result = tc.coherence(random_words, corpus_dict, epsilon)
    tc_results.append(result)

ofile.write("AVG: " + str(numpy.average(tc_results))+"\n")
ofile.write("SD: "+ str(numpy.std(tc_results))+"\n\n")
for item in tc_results:
    ofile.write(str(item)+"\n")

for item in words_list:
    ofile.write(str(item)+"\n")

