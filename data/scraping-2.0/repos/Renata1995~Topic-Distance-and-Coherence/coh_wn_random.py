import random
import sys

import numpy
from gensim import corpora

from coherence.wn import WordNetEvaluator
from topic.topic import Topic
from nltk.corpus import wordnet as wn
from nltk.corpus import reuters
from nltk.corpus import brown
# python random_tc.py <dname> <word_count> <sample_times> <output>
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
    tcmethod = "path"
else:
    tcmethod = sys.argv[4]
    print tcmethod

if len(sys.argv) <= 5:
    ic = False
else:
    if sys.argv[5] == "ic":
        ic = True
    else:
        ic = False


dictionary = corpora.Dictionary.load(dname + "/dict.dict")
print "Load dictionary",
print dictionary
corpus_fname = dname + '/bow_corpus.mm'
print "Load Corpus File " + corpus_fname
corpus = corpora.MmCorpus(corpus_fname)

# transfer each doc in the corpus into a dictionary
corpus_dict = []
for doc in corpus:
    corpus_dict.append(dict(doc))
dictlen = len(dictionary)

tc = WordNetEvaluator()

tc_means = []
tc_medians = []
words_list = []

ofilemean = open(dname + "/"+tcmethod+"_mean_rand_"+str(word_count)+".txt", "w")
ofilemedian = open(dname + "/"+tcmethod+"_median_rand_"+str(word_count)+".txt", "w")

if ic:
    if dname == "reuters_LDA":
        src_ic = wn.ic(reuters, False, 0.0)
    else:
        src_ic = wn.ic(brown, False, 0.0)



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

    randt = Topic()
    for key in keylist:
        randt.add((key, 0.1))

    # calculate topic coherence based on randomly generated words
    if ic:
        result = tc.evaluate_ic(randt, word_count, src_ic, tcmethod, not_write=True)
    else:
        result = tc.evaluate(randt, word_count, tcmethod, not_write=True)

    if (not numpy.isnan(result[1])) and result[1] < 10000:
        rmean = result[1]
    else:
        rmean = 0.0

    if (not numpy.isnan(result[2])) and result[1] < 10000:
        rmedian = result[2]
    else:
        rmedian = 0.0
        
    tc_means.append(rmean)
    tc_medians.append(rmedian)

ofilemean.write("Mean: " + str(numpy.mean(tc_means)) + "\n")
ofilemean.write("SD: " + str(numpy.std(tc_means)) + "\n\n")
for item in tc_means:
    ofilemean.write(str(item) + "\n")

for item in words_list:
    ofilemean.write(str(item) + "\n")

ofilemedian.write("Mean: " + str(numpy.mean(tc_medians)) + "\n")
ofilemedian.write("SD: " + str(numpy.std(tc_medians)) + "\n\n")
for item in tc_medians:
    ofilemedian.write(str(item) + "\n")

for item in words_list:
    ofilemedian.write(str(item) + "\n")

