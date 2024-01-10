from coherence.wn import WordNetEvaluator
import sys
import utils.name_convention as name
from topic.topicio import TopicIO
from nltk.corpus import wordnet as wn
from nltk.corpus import reuters
from nltk.corpus import brown

# syntax: python  coh_wn_read.py <corpus type> <# of topics> <src> <word count>
#  <corpus type> default to bag of words. b for binary, t for tf-idf, anything else or missing for bag of words
#  <# of topics> number of topics. default to 8
#  <src> src folder which contains documents for LDA
#  <wordnet method> default to path
#  <need_ic> whether the wordnet method needs ic or not. "ic": ic is needed    anything else: ic is not needed.
#  <word count> the number of top words used in the calculation of topic coherence
#  <startw> the start point of collecting words

if len(sys.argv) <= 1:
    corpus_type = "bow"
else:
    if sys.argv[1] == "t":
        corpus_type = "tfidf"
    elif sys.argv[1] == "b":
        corpus_type = "binary"
    else:
        corpus_type = "bow"

if len(sys.argv) <= 2:
    topics_count = 3
else:
    topics_count = int(sys.argv[2])

if len(sys.argv) <= 3:
    src = "pp_reuters"
else:
    src = sys.argv[3]

if len(sys.argv) <= 4:
    tc = "path"
else:
    tc = sys.argv[4]

if len(sys.argv) <= 5:
    need_ic = False
else:
    if sys.argv[5] == "ic":
        need_ic = True
    else:
        need_ic = False

if len(sys.argv) <= 6:
    words_count = 10
else:
    words_count = int(sys.argv[6])

if len(sys.argv) <= 7:
    startw = 0
else:
    startw = int(sys.argv[7])

dname = name.get_output_dir(corpus_type, topics_count, src)

# read topics
tio = TopicIO()
tlist = tio.read_topics(dname + name.topics_dir())

# generate te file name
fname = dname + name.te_preprocess(tc, words_count, startw)
prefile = open(fname, "w")

zerofile = dname + "/zeros_" + tc + "_w" + str(words_count) + ".txt"
zerofile = open(zerofile, "w")

# calculate topic evaluation values
tclist = []
te = WordNetEvaluator()
if not need_ic:
    for index, topic in enumerate(tlist):
        tclist.append([index, te.evaluate(topic, words_count, tc, prefile, zerofile, startw=startw)])
else:
    if src == "pp_reuters":
        src_ic = wn.ic(reuters, False, 0.0)
    elif src == "pp_brown":
        src_ic = wn.ic(brown, False, 0.0)
    else:
        print "No inoformation content for this corpus"
        print "Create information content first"

    for index, topic in enumerate(tlist):
        tclist.append([index, te.evaluate_ic(topic, words_count, src_ic, tc, prefile, zerofile, startw=startw)])
