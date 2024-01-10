import sys

from gensim import corpora

from topic.topicio import TopicIO
from coherence.umass import TopicCoherence
import utils.name_convention as name
from utils.corpus import arg_tfidf

#
# syntax: python  tcReader.py <input directory name> <corpus type> <# of topics> <src> <word count>
#  <dictionary name> the name of the input dictionary
#  <corpus type> default to bag of words. b for binary, t for tf-idf, anything else or missing for bag of words
#  <# of topics> number of topics. default to 8
#  <src> src folder which contains documents for LDA
#  <max_wc> the number of top words used in the calculation of topic coherence
#  <startw> the start point of collecting words
#  <tfidf> t: use tfidf topic coherence measure   anything else or missing: use regular topic coherence measure
#

#
# Read command line parameters
#
if len(sys.argv) <= 1:
    dname = 'pp_test_LDA'
else:
    dname = sys.argv[1]

if len(sys.argv) <= 2:
    corpus_type = "bow"
else:
    if sys.argv[2] == "t":
        corpus_type = "tfidf"
    elif sys.argv[2] == "b":
        corpus_type = "binary"
    else:
        corpus_type = "bow"

if len(sys.argv) <= 3:
    topics_count = 8;
else:
    topics_count = int(sys.argv[3]);

if len(sys.argv) <= 4:
    src = "pp_test"
else:
    src = sys.argv[4]

if len(sys.argv) <= 5:
    max_wc = 10
else:
    max_wc = int(sys.argv[5])

if len(sys.argv) <= 6:
    startw = 0
else:
    startw = int(sys.argv[6])

if len(sys.argv) <= 7:
    tfidf = False
else:
    if sys.argv[7] == "t":
        tfidf = True
    else:
        tfidf = False

output = name.get_output_dir(corpus_type, topics_count, src)

print "input directory : " + dname
print "corpus type :" + corpus_type
print "# of topics : " + str(topics_count)
print "src : " + src
print "# of words used for topic coherence: " + str(max_wc)
print "start word :" + str(startw)
print "tfidf : " + str(tfidf)
print "output : " + output
print "\n"

# Load directory
dictionary = corpora.Dictionary.load(dname + "/dict.dict")
print(dictionary)

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

# Transfer each doc(list) in the corpus into a dictionary
corpus_dict = []
for doc in corpus:
    corpus_dict.append(dict(doc))

# Init all helpers
topics_io = TopicIO()
tc = TopicCoherence()

# get all topics
tlist = topics_io.read_topics(output + "/topics")

# sort all words by decreasing frequency
tlist2 = []
for topic in tlist:
    topic.sort()
    tlist2.append(topic.list_words(max_wc, start=startw))

# construct a dictionary that contains top startw - max_wc words and their ids in each topic
word_id_dict = {}
for topic in tlist2:
    for word in topic:
        if word not in word_id_dict.keys():  # check whether the key already exists
            wordkey = -1
            for key, value in dictionary.iteritems():  # key-id value-word
                if dictionary.get(key) == word:
                    wordkey = key
                    break
            if wordkey > -1:
                word_id_dict[word] = wordkey

# id list
key_list = [value for key, value in word_id_dict.iteritems()]

# prepare output file
tf_file = name.tc_tf_file(dname, corpus_type, topics_count, startw, tfidf)
co_occur_file = name.tc_co_occur_file(dname, corpus_type, topics_count, startw, tfidf)

# write term frequency and words co-occurrence frequency to files
tf_dict = tc.word_list_doc_freq(key_list, corpus_dict, dictionary)
tc.write_dict(tf_dict, tf_file)
co_occur_dict = tc.words_cooccur(key_list, corpus_dict, dictionary)
tc.write_dict(co_occur_dict, co_occur_file)
