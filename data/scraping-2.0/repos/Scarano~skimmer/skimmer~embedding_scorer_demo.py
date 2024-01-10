import argparse
import os
import tempfile
import webbrowser
from typing import IO

import joblib
import numpy as np
import sys

from skimmer.embedding_scorer import EmbeddingScorer, Method
from skimmer.jina_local_embedding import JinaLocalEmbedding
from skimmer.openai_embedding import OpenAIEmbedding
from skimmer.openai_summarizer import OpenAISummarizer
from skimmer.parser import RightBranchingParser

from skimmer.span_scorer import ScoredSpan


def scored_spans_as_html(doc: str, spans: list[ScoredSpan], f: IO):
    scores = np.array([span.score for span in spans])
    colors = np.array([[255, 127, 127],   # Red for lowest score
                       [255, 255, 255],   # White for median
                       [127, 255, 127]])  # Green for highest
    min_score = np.min(scores)
    median_score = np.median(scores)
    max_score = np.max(scores)
    for i, span in enumerate(spans):
        if i > 0:
            f.write(doc[spans[i-1].end:span.start])
        rgb = [str(int(np.interp(scores[i], [min_score, median_score, max_score], col)))
               for col in colors.T]
        f.write(f'<span style="background-color: rgb({",".join(rgb)});">')
        f.write(f'<br>[{span.score:.3f}] {doc[span.start:span.end]}')
        f.write('</span>')


def score_to_html(doc, method, cache_dir, chunk_size, length_penalty, summary_override):
    """
    Do a demo run of the EmbeddingScorer on a single doc.
    :param doc: Document to abridge
    :param method: Method to use for abridging
    :param chunk_size: Number of sentences to abridge as a unit
    :param length_penalty: Scores are divided by sentence lenght to this power
    :param summary_override: If not None, this string will be used as the summary instead of
        generating one. Only for testing purposes. And only for when doc has fewer than
        `chunk_size` sentences.
    """
    if cache_dir is not None:
        embed_memory = joblib.Memory(os.path.join(cache_dir, 'embedding_cache_openai'),
                                     mmap_mode='c', verbose=0)
        summarize_memory = joblib.Memory(os.path.join(cache_dir, 'summary_cache_openai'),
                                         mmap_mode='c', verbose=0)
    else:
        embed_memory = None
        summarize_memory = None
    # parser = Parser('en')
    parser = RightBranchingParser('en')
    embed = OpenAIEmbedding(memory=embed_memory)
    # embed = JinaLocalEmbedding(JinaLocalEmbedding.SMALL_MODEL, embed_memory)
    if summary_override:
        summarize = lambda _: summary_override
    else:
        summarize = OpenAISummarizer(memory=summarize_memory)
    scorer = EmbeddingScorer(method, chunk_size, length_penalty, parser, embed, summarize)
    spans = scorer(doc)
    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as f:
        print(f"Saving HTML output to: {f.name}")
        scored_spans_as_html(doc, spans, f)
        webbrowser.open('file://' + f.name, new=2)
        # input('Press enter to complete and delete temp file.')


def demo():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Scorer')
    parser.add_argument('doc', type=str, help='Path to document to abridge')
    parser.add_argument('--method', type=str, default=Method.default().value,
                         choices=[m.value for m in Method],
                         help='Method to use for abridging')
    parser.add_argument('--chunk-size', type=int, default=20,
                        help='Maximum number of sentences to abridge at once')
    parser.add_argument('--length-penalty', type=float, default=0.0)
    parser.add_argument('--summary', type=str, default=None,
                        help='Path to target summary used to override generates summary (for testing)')
    parser.add_argument('--cache-dir', type=str)
    args = parser.parse_args()

    method = Method.of(parser.parse_args().method)

    with open(args.doc) as f:
        doc = f.read()
    if args.summary is None:
        summary = None
    else:
        with open(args.summary) as f:
            summary = f.read()
    score_to_html(doc, method, args.cache_dir,
                  args.chunk_size, length_penalty=args.length_penalty,
                  summary_override=summary)

    return 0


if __name__=='__main__':
    sys.exit(demo())

