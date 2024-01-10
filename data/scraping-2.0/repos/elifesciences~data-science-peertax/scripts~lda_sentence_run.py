#!/usr/bin/env python3

import argparse
import json
import os
import logging
from functools import singledispatch
from typing import List

import numpy as np
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis.gensim


LOGGER = logging.getLogger(__name__)


class EnvrionmentVariables:
    INPUT_FILE = 'INPUT_FILE'
    OUTPUT_DIRECTORY = 'OUTPUT_DIRECTORY'
    LDA_NUM_TOPICS = 'LDA_NUM_TOPICS'
    LIMIT = 'LIMIT'
    LDA_PASSES = 'LDA_PASSES'
    LDA_ITERATIONS = 'LDA_ITERATIONS'
    LDA_EVAL_EVERY = 'LDA_EVAL_EVERY'


@singledispatch
def to_serializable(val):
    return val


@to_serializable.register(np.float32)
def _serializable_float32(val):
    return float(val)


def _add_required_argument(parser: argparse.ArgumentParser, *args, default, **kwargs):
    parser.add_argument(*args, required=default == '', default=default, **kwargs)


def _to_int_or_none(value: str) -> int:
    if not value:
        return None
    return int(value)


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LDA Sentence Run"
    )

    _add_required_argument(
        parser,
        "--input-file",
        default=os.environ.get(EnvrionmentVariables.INPUT_FILE, ''),
        help="Input file to use (TSV)"
    )

    _add_required_argument(
        parser,
        "--output-directory",
        default=os.environ.get(EnvrionmentVariables.OUTPUT_DIRECTORY, ''),
        help="Output directory"
    )

    _add_required_argument(
        parser,
        "--lda-num-topics",
        default=os.environ.get(EnvrionmentVariables.LDA_NUM_TOPICS, ''),
        type=int,
        help="LDA number of topics"
    )

    parser.add_argument(
        '--lda-passes',
        default=os.environ.get(EnvrionmentVariables.LDA_PASSES, '30'),
        type=int,
        help='Number of LDA passes'
    )

    parser.add_argument(
        '--lda-iterations',
        default=os.environ.get(EnvrionmentVariables.LDA_ITERATIONS, '1000'),
        type=int,
        help='Number of LDA iterations'
    )

    parser.add_argument(
        '--lda-eval-every',
        default=os.environ.get(EnvrionmentVariables.LDA_EVAL_EVERY, '10'),
        type=int,
        help='Eval after every specified iterations'
    )

    parser.add_argument(
        '--limit',
        default=os.environ.get(EnvrionmentVariables.LIMIT),
        type=_to_int_or_none,
        help='limit number of rows to read from input file (mainly for testing)'
    )

    args = parser.parse_args(argv)
    return args


def read_tokens_list(filename: str, limit: int = None) -> pd.DataFrame:
    df = pd.read_csv(filename, sep='\t', nrows=limit)
    LOGGER.info('df: %s', df.head())
    return df['token'].apply(eval)


def run(args: argparse.Namespace):
    output_directory = args.output_directory
    num_topics = int(args.lda_num_topics)

    model_meta = {}

    model_meta['limit'] = args.limit

    tokens_list = read_tokens_list(args.input_file, limit=args.limit)

    # Create Dictionary
    id2word = corpora.Dictionary(tokens_list)

    model_meta['num_unique_tokens_before_filter'] = len(id2word)

    token_filter_params = dict(no_below=20, no_above=0.75)
    model_meta['token_filter_params'] = token_filter_params

    LOGGER.info('Dict size (no filter): %d', len(id2word))
    id2word.filter_extremes(**token_filter_params)
    LOGGER.info('Dict size (after filter): %d', len(id2word))

    # Term Document Frequency
    corpus = [id2word.doc2bow(tokens) for tokens in tokens_list]
    LOGGER.info('Number of unique tokens: %d', len(id2word))
    LOGGER.info('Number of documents: %d', len(corpus))

    model_meta['num_unique_tokens'] = len(id2word)
    model_meta['num_documents'] = len(corpus)

    # Set training parameters.
    lda_params = dict(
        num_topics=num_topics,
        random_state=100,
        eval_every=int(args.lda_eval_every),
        passes=int(args.lda_passes),
        iterations=int(args.lda_iterations),
        per_word_topics=True
    )

    model_meta['lda_params'] = lda_params

    lda_model = gensim.models.ldamulticore.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        **lda_params
    )

    # Save model
    output_filename_prefix = os.path.join(output_directory, 'Model_%d' % num_topics)
    lda_model.save(output_filename_prefix)
    LOGGER.info('Model saved at location %s', output_filename_prefix)

    # Compute complexity score of model
    model_meta['perplexity'] = lda_model.log_perplexity(corpus)
    # Compute coherence scores (c_v and umass)
    cv_model_lda = CoherenceModel(
        model=lda_model, texts=tokens_list, dictionary=id2word, coherence='c_v'
    )
    cv_lda = cv_model_lda.get_coherence()
    umass_model_lda = CoherenceModel(
        model=lda_model, texts=tokens_list, dictionary=id2word, coherence="u_mass"
    )
    umass_lda = umass_model_lda.get_coherence()
    model_meta['cv_lda'] = cv_lda
    model_meta['umass_lda'] = umass_lda

    # Get top topics and average topic coherence
    model_meta['top_topics'] = lda_model.top_topics(corpus)

    LOGGER.info('model_meta: %s', model_meta)
    with open(output_filename_prefix + '-meta.json', 'w') as fp:
        json.dump(model_meta, fp, indent=4, default=to_serializable)

    # Save visualization as an html file
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, sort_topics=False)
    file_vis = output_filename_prefix + '.html'
    pyLDAvis.save_html(vis, file_vis)


def main(argv: List[str] = None):
    LOGGER.info('lda sentence run: %s', argv)
    LOGGER.debug('env: %s', dict(os.environ))

    args = parse_args(argv)
    LOGGER.info('args: %s', args)
    run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    main()
