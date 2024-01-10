import os

import joblib

from skimmer import embedding_scorer
from skimmer import summary_matching_scorer
from skimmer.abridger import Abridger
from skimmer.embedding_scorer import EmbeddingScorer
from skimmer.lead_baseline_scorer import LeadBaselineScorer
from skimmer.summary_matching_scorer import SummaryMatchingScorer, SummaryMatchingClauseScorer
from skimmer.parser import RightBranchingParser, StanzaParser


CONFIG_METHOD = 'method'
CONFIG_METHOD_LEAD_BASELINE = 'lead-baseline'

CONFIG_EMBEDDING = 'embedding'
CONFIG_EMBEDDING_OPENAI = 'openai'
CONFIG_EMBEDDING_JINA_LOCAL_SMALL = 'jina-local-small'

CONFIG_SUMMARY = 'summary'
CONFIG_SUMMARY_OPENAI = 'openai'

CONFIG_SUMMARY_PROMPT = 'summary-prompt'

CONFIG_CHUNK_SIZE = 'chunk-size'
CONFIG_CHUNK_SIZE_DEFAULT = 30

CONFIG_LENGTH_PENALTY = 'length-penalty'
CONFIG_LENGTH_PENALTY_DEFAULT = 1.0

CONFIG_ABRIDGE_THRESHOLD = 'abridge-threshold'
CONFIG_ABRIDGE_THRESHOLD_DEFAULT = 0.5

CONFIG_ABRIDGE_MAX_SENTENCES = 'abridge-max-sentences'


class InvalidConfigException(Exception):
    def __init__(self, config, key):
        super().__init__(f"Invalid configuration {key}: {config.get(key, '(unspecified)')}")


def build_scorer_from_config(config: dict, work_dir: str):

    if CONFIG_EMBEDDING in config:
        embedding_cache_path = os.path.join(work_dir, f'embedding_cache_{config[CONFIG_EMBEDDING]}')
        embed_memory = joblib.Memory(embedding_cache_path, mmap_mode='c', verbose=0)
    else:
        embed_memory = None

    if config.get(CONFIG_EMBEDDING) == CONFIG_EMBEDDING_OPENAI:
        from skimmer.openai_embedding import OpenAIEmbedding
        embed = OpenAIEmbedding(memory=embed_memory)
    elif config.get(CONFIG_EMBEDDING) == CONFIG_EMBEDDING_JINA_LOCAL_SMALL:
        from skimmer.jina_local_embedding import JinaLocalEmbedding
        embed = JinaLocalEmbedding(JinaLocalEmbedding.SMALL_MODEL, embed_memory)
    else:
        embed = None

    if config.get(CONFIG_SUMMARY, '') == '':
        summarize = None
    else:
        summary_cache_path = os.path.join(work_dir, f'summary_cache_{config[CONFIG_SUMMARY]}')
        summary_memory = joblib.Memory(summary_cache_path, mmap_mode='c', verbose=0)

        summary_prompt = config[CONFIG_SUMMARY_PROMPT]
        if config[CONFIG_SUMMARY] == CONFIG_SUMMARY_OPENAI:
            from skimmer.openai_summarizer import OpenAISummarizer
            summarize = OpenAISummarizer(prompt_name=summary_prompt, memory=summary_memory)
        else:
            raise InvalidConfigException(config, CONFIG_SUMMARY)

    method_str = config[CONFIG_METHOD]

    length_penalty = config.get(CONFIG_LENGTH_PENALTY, CONFIG_LENGTH_PENALTY_DEFAULT)

    if method_str == CONFIG_METHOD_LEAD_BASELINE:
        parser = RightBranchingParser('en')
        return LeadBaselineScorer(parser)

    elif embedding_scorer.Method.contains_value(method_str):
        method = embedding_scorer.Method.of(method_str)

        parser = RightBranchingParser('en')

        chunk_size = config.get(CONFIG_CHUNK_SIZE, CONFIG_CHUNK_SIZE_DEFAULT)

        return EmbeddingScorer(method, chunk_size, length_penalty, parser, embed, summarize)

    elif summary_matching_scorer.Method.contains_value(method_str):
        method = summary_matching_scorer.Method.of(method_str)

        match method:
            case summary_matching_scorer.Method.SENTENCE_SUMMARY_MATCHING:

                parser = RightBranchingParser('en')
                return SummaryMatchingScorer(parser, embed, summarize)

            case summary_matching_scorer.Method.CLAUSE_SUMMARY_MATCHING:

                parse_cache_path = os.path.join(work_dir, f'parse_cache')
                parse_memory = joblib.Memory(parse_cache_path, mmap_mode='c', verbose=0)
                parser = StanzaParser('en', parse_memory)
                return SummaryMatchingClauseScorer(parser, embed, summarize, length_penalty)

            case _:
                raise Exception("bug")
    else:
        raise InvalidConfigException(config, CONFIG_METHOD)


def build_abridger_from_config(config, work_dir):
    scorer = build_scorer_from_config(config, work_dir)
    threshold = config.get(CONFIG_ABRIDGE_THRESHOLD, CONFIG_ABRIDGE_THRESHOLD_DEFAULT)
    max_sentences = config.get(CONFIG_ABRIDGE_MAX_SENTENCES, None)
    return Abridger(scorer, threshold, max_sentences)
