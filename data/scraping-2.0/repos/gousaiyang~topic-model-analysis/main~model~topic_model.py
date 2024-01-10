import logging
import os
import re
import sys
import time

import colorlabels as cl
import pyLDAvis
import pyLDAvis.gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore

from util import (NoConsoleOutput, TimeMeasure, data_source_file,
                  file_read_json, file_write_json, log_file, model_file,
                  open_html_in_browser, report_file)

N_WORKERS = os.cpu_count()


def measure_coherence(model, texts, corpus, dictionary):
    cm = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary,
                        coherence='u_mass')
    u_mass = cm.get_coherence()

    cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary,
                        coherence='c_v')
    c_v = cm.get_coherence()

    cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary,
                        coherence='c_uci')
    c_uci = cm.get_coherence()

    cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary,
                        coherence='c_npmi')
    c_npmi = cm.get_coherence()

    cl.info('Topic coherence: u_mass = %f, c_v = %f, c_uci = %f, c_npmi = %f'
            % (u_mass, c_v, c_uci, c_npmi))


def lda_topic_model(input_filename, keyword, size, *, num_topics,
                    iterations=50, passes=1, chunksize=2000, eval_every=10,
                    verbose=False, gamma_threshold=0.001, filter_no_below=5,
                    filter_no_above=0.5, filter_keep_n=100000,
                    open_browser=True):
    cl.section('LDA Topic Model Training')
    cl.info('Keyword: %s' % keyword)
    cl.info('Data size: %d' % size)
    cl.info('Number of topics: %d' % num_topics)
    cl.info('Iterations: %d' % iterations)
    cl.info('Passes: %d' % passes)
    cl.info('Chunk size: %d' % chunksize)
    cl.info('Eval every: %s' % eval_every)
    cl.info('Verbose: %s' % verbose)
    cl.info('Gamma Threshold: %f' % gamma_threshold)
    cl.info('Filter no below: %d' % filter_no_below)
    cl.info('Filter no above: %f' % filter_no_above)
    cl.info('Filter keep n: %d' % filter_keep_n)

    assert re.fullmatch(r'[-_0-9a-zA-Z+]+', keyword)

    input_filename = data_source_file(input_filename)
    description = '%s-%d-%d-%dx%d-%s' % (keyword, size, num_topics, iterations,
                                         passes, time.strftime('%Y%m%d%H%M%S'))

    if verbose:
        log_filename = log_file('ldalog-%s.log' % description)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.DEBUG, filename=log_filename)
        cl.info('Writing logs into file: %s' % log_filename)

    with TimeMeasure('load_preprocessed_text'):
        preprocessed_texts = file_read_json(input_filename)
        preprocessed_texts = [item[1] for item in preprocessed_texts]

    with TimeMeasure('gen_dict_corpus'):
        cl.progress('Generating dictionary and corpus...')

        dictionary = Dictionary(preprocessed_texts, prune_at=None)
        dictionary.filter_extremes(no_below=filter_no_below,
                                   no_above=filter_no_above,
                                   keep_n=filter_keep_n)
        dictionary.compactify()

        corpus = [dictionary.doc2bow(text) for text in preprocessed_texts]

        corpusfilename = model_file('ldacorpus-%s.json' % description)
        file_write_json(corpusfilename, corpus)
        cl.success('Corpus saved as: %s' % corpusfilename)

    with TimeMeasure('training'):
        cl.progress('Performing training...')

        with NoConsoleOutput():
            ldamodel = LdaMulticore(corpus, workers=N_WORKERS,
                                    id2word=dictionary, num_topics=num_topics,
                                    iterations=iterations, passes=passes,
                                    chunksize=chunksize, eval_every=eval_every,
                                    gamma_threshold=gamma_threshold,
                                    alpha='symmetric', eta='auto')

        cl.success('Training finished.')

    with TimeMeasure('save_model'):
        modelfilename = 'ldamodel-%s' % description
        ldamodel.save(model_file(modelfilename))
        cl.success('Model saved as: %s' % modelfilename)

    with TimeMeasure('measure_coherence'):
        cl.progress('Measuring topic coherence...')
        measure_coherence(ldamodel, preprocessed_texts, corpus, dictionary)

    with TimeMeasure('vis_save'):
        cl.progress('Preparing visualization...')
        vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
        htmlfilename = 'ldavis-%s.html' % description
        htmlfilename = report_file(htmlfilename)
        pyLDAvis.save_html(vis, htmlfilename)
        cl.success('Visualized result saved in file: %s' % htmlfilename)

    if open_browser:
        open_html_in_browser(htmlfilename)
