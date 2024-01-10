import sys

from gensim import corpora

from topic.topicio import TopicIO
from coherence.umass import TopicCoherence
import utils.name_convention as name

#
# syntax: python  tc_read.py <input directory name> <corpus type> <# of topics> <src> <word count>
#  <input dictionary name> the directory that has the required corpus
#  <corpus type> default to bag of words. b for binary, t for tf-idf, anything else or missing for bag of words
#  <# of topics> number of topics. default to 8
#  <src> src folder which contains documents for LDA
#  <word count> the number of top words used in the calculation of topic coherence
#  <startw> the start point of collecting words
#  <tfidf> t: use tfidf topic coherence measure   anything else or missing: use regular topic coherence measure

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
    word_count = 10
else:
    word_count = int(sys.argv[5])

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

if tfidf:
    epsilon = 0.0001
else:
    epsilon = 1

output = name.get_output_dir(corpus_type, topics_count, src)

print "input directory : " + dname
print "corpus type :" + corpus_type
print "# of topics : " + str(topics_count)
print "src : " + src
print "# of words used for topic coherence: " + str(word_count)
print "output : " + output
print "word count : " + str(word_count)
print "startw : " + str(startw)
print "Tfidf : " + str(tfidf)
print "\n"

# Load directory
dictionary = corpora.Dictionary.load(dname + "/dict.dict")
print(dictionary)

# Init helpers
topics_io = TopicIO()
tc = TopicCoherence()

# get all topics
tlist = topics_io.read_topics(output + "/topics")

# sort all words by decreasing frequency
tlist2 = []
for topic in tlist:
    topic.sort()
    tlist2.append(topic.list(word_count, start=startw))

# prepare output file
tf_file = name.tc_tf_file(dname, corpus_type, topics_count, startw, tfidf)
co_occur_file = name.tc_co_occur_file(dname, corpus_type, topics_count, startw, tfidf)

wd_dict = tc.read_into_dict(tf_file)
cofreq_dict = tc.read_into_dict(co_occur_file)


# calculate topic coherence values for each topic with a specific number of words
ofilename = name.tc_contribution(output, word_count, startw, tfidf)
ofile = open(ofilename, "w")
ctlist = []
for index, t in enumerate(tlist2):
    t = t[:word_count]
    subt = [wt[0] for wt in t]
    ofile.write("topic " + str(index) + "\n")
    ctlist.append((index, tc.coherence_dict(subt, wd_dict, cofreq_dict, ofile, epsilon=epsilon), t))
    ofile.write("\n")

# sort all topics by topic coherence
ctlist = list(reversed(sorted(ctlist, key=lambda x: x[1])))

ofilename = name.tc_output_file(output, word_count, startw, tfidf)
ofile = open(ofilename, "w")
for tctuple in ctlist:
    ofile.write("topic  " + str(tctuple[0]) + "   " + str(tctuple[1]) + "\n\n")
    for item in tctuple[2]:
        ofile.write(item[0] + " : " + str(item[1]) + "\n")
    ofile.write("\n\n")
