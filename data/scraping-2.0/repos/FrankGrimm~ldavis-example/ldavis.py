import argparse
import os, os.path
import signal
import sys
import threading
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s')

from http.server import HTTPServer, SimpleHTTPRequestHandler

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
from glob import glob
from gensim.models import CoherenceModel
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def parse_args():
    p = argparse.ArgumentParser(description="LDA and visualization server")

    p.add_argument("--bind", type=str, default="127.0.0.1", help="host to bind to")
    p.add_argument("--port", type=int, default=5050, help="port to bind to")

    p.add_argument("--datadir", type=str, default="./data/", help="data directory location")
    p.add_argument("--nobrowser", action="store_true")
    p.add_argument("--debug", action="store_true")

    p.add_argument("--numtopics", type=int, default=50, help="LDA topics")
    p.add_argument("--passes", type=int, default=5, help="LDA passes")
    p.add_argument("--chunksize", type=int, default=1000, help="LDA chunk size")

    return p.parse_args()

args = None
webthread = None

def load_datasets():
    ds = {}
    pattern = os.path.join("./corpora/*.txt")

    for corpus_file in glob(pattern):
        cur_dataset = {'id': ".".join(os.path.basename(corpus_file).split(".")[:-1]), \
                'corpus': os.path.abspath(corpus_file)}

        ds[cur_dataset['id']] = cur_dataset

    return ds

def load_corpus(datasets, dataset_id):
    corpus_file = datasets[dataset_id].get("corpus", None)
    if corpus_file is None:
        raise Exception("failed to find corpus file for dataset %s" % dataset_id)

    print("[open] %s" % corpus_file)
    with open(corpus_file, "rt") as infile:
        for lineidx, line in enumerate(infile):
            if lineidx > 0 and lineidx % 10000 == 0:
                print("[progress] %s %s" % (corpus_file, lineidx))
            line = line.rstrip()
            delim = '\t'
            if line.strip() == ' ':
                continue
            if not delim in line:
                delim = " "
            sample_id, line = line.split(delim, 1)
            sample_id = sample_id.strip()
            line = line.strip()
            yield (sample_id, line)

def perform_lda(datasets, dataset_id, options):

    additional_stopwords = map(str.strip, map(str, options.get("additional_stopwords", [])))
    additional_stopwords = list(filter(lambda w: not w is None and w != "", additional_stopwords))

    sw = stopwords.words("english")
    if len(additional_stopwords):
        sw.extend(additional_stopwords)

    print("[stopwords] %s" % ",".join(sw))

    pos_filter = options.get("pos_filter", ['NOUN', 'ADJ', 'VERB', 'ADV'])

    full_corpus = []

    for sample_id, doc in load_corpus(datasets, dataset_id):
        # preprocess document
        doc = doc.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
        doc = doc.replace("\t", " ")
        for remove_char in []:
            doc = doc.replace(remove_char, "")
        doc = " ".join([w for w in doc.strip().split(" ") if not w.strip() == ''])

        # accent removal, stopwords, and length filtering
        deacc = options.get("deacc", True)
        min_token_len = options.get("min_token_len", 2)
        max_token_len = options.get("max_token_len", 15)
        doc = [ w for w in gensim.utils.simple_preprocess(doc, deacc=deacc, \
                min_len=min_token_len, max_len=max_token_len) \
                if not w in sw]

        lemmatize = options.get("lemmatize", True)
        if lemmatize or len(pos_filter) > 0:
            nlpdoc = nlp(" ".join(doc))

            doc = [ token.text if not lemmatize else token.lemma_ \
                    for token in nlpdoc \
                    if len(pos_filter) == 0 or token.pos_ in pos_filter ]
            doc = list(map(str.strip, doc))

        full_corpus.append(doc)

    # dictionary
    dict_file = os.path.abspath(os.path.join("./lda_models/", "%s.dict" % dataset_id))
    datasets[dataset_id]['dict'] = dict_file
    if not os.path.exists(dict_file):
        id2word = corpora.Dictionary(full_corpus)
        id2word.save_as_text(dict_file)
        print("[dictionary] created %s" % dict_file)
    id2word = corpora.Dictionary.load_from_text(dict_file)
    print("[dictionary] %s" % dict_file)

    full_corpus_texts = full_corpus
    full_corpus = list(map(id2word.doc2bow, full_corpus))

    model_filename = os.path.abspath(os.path.join("./lda_models/", "%s.model" % dataset_id))
    if not os.path.exists(model_filename):
        print("[lda model] creating %s" % model_filename)
        lda_model = gensim.models.ldamodel.LdaModel(corpus=full_corpus,
                id2word=id2word,
                num_topics=args.numtopics,
                random_state=100,
                update_every=1,
                chunksize=args.chunksize,
                passes=args.passes,
                alpha='auto',
                per_word_topics=True)
        lda_model.save(model_filename)

    print("[lda model] loading %s" % model_filename)
    lda_model = gensim.models.ldamodel.LdaModel.load(model_filename)

    lda_model.print_topics(num_topics = 10, num_words = 5)

    print()
    print("[perplexity] dataset %s: %s" % (\
            dataset_id, \
            lda_model.log_perplexity(full_corpus)))

    coherence_model_lda = CoherenceModel(model=lda_model, texts=full_corpus_texts, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print("[coherence] dataset %s: %s" % (\
            dataset_id, coherence_lda))

    vis_filename = os.path.abspath(os.path.join("./vis/", "%s.html" % dataset_id))
    if not os.path.exists(vis_filename):
        print("[vis] creating %s" % vis_filename)
        p = pyLDAvis.gensim.prepare(lda_model, full_corpus, id2word)
        pyLDAvis.save_html(p, vis_filename)

    print("[vis] %s" % vis_filename)

def run_application():
    server = HTTPServer((args.bind or '', args.port), SimpleHTTPRequestHandler)
    server.serve_forever()

def start_server():
    global args, webthread

    args = parse_args()

    if not os.path.exists(args.datadir):
        print("could not find data directory at %s, make sure it exists" % \
                os.path.abspath(args.datadir))
        sys.exit(2)

    print("[data directory] %s" % args.datadir)

    os.chdir(os.path.abspath(args.datadir))
    uri = "http://%s:%s" % (args.bind, args.port)
    print("starting server at %s" % uri)

    webthread = threading.Thread(target=run_application)
    webthread.daemon=True
    webthread.start()

    if not args.nobrowser:
        import webbrowser
        webbrowser.open_new_tab(uri)

    all_datasets = load_datasets()
    print("[datasets] %s (%s)" % (", ".join(all_datasets.keys()), len(all_datasets)))

    lda_options = {}

    for ds_id, dataset in all_datasets.items():
        print("[dataset]", ds_id, dataset)
        perform_lda(all_datasets, ds_id, lda_options)

    try:
        signal.pause()
    except (KeyboardInterrupt, SystemExit):
        print("[exit] requested")

if __name__ == "__main__":
    start_server()
