import logging
from itertools import islice
from typing import List

import numpy as np
import pandas as pd
import tiktoken
from tqdm import tqdm

from ..console import verbose_print
from ..file_handler.abstract_handler import ParsedCode
from ..openai_service import OpenAIService, tokens_from_string

logger = logging.getLogger(__name__)


class CodeProcessor:
    def __init__(self, code_root, openai_service: OpenAIService = None):
        # Todo: add code root
        self.code_root = code_root
        self.openai_service = openai_service if openai_service else OpenAIService()

    def process(self, code_blocks: List[ParsedCode]):
        if len(code_blocks) == 0:
            logger.verbose_info("No code blocks to process")
            return None
        df = pd.DataFrame(code_blocks)
        logger.verbose_info(
            f"Generating openai embeddings for {len(df)} code blocks. This may take a while because of rate limiting..."
        )

        def len_safe_get_embedding(text):
            max_tokens = 8191
            encoding_name = "cl100k_base"

            chunk_embeddings = []
            chunk_lens = []
            for chunk in CodeProcessor._chunked_tokens(
                text, encoding_name=encoding_name, chunk_length=max_tokens
            ):
                chunk_embeddings.append(self.openai_service.get_embedding(chunk))
                chunk_lens.append(len(chunk))

            chunk_embedding = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
            return chunk_embedding / np.linalg.norm(
                chunk_embedding
            )  # normalizes length to 1

        if logger.getEffectiveLevel() < logging.INFO:
            tqdm.pandas()
            df["code_embedding"] = df["code"].progress_apply(len_safe_get_embedding)
        else:
            df["code_embedding"] = df["code"].apply(len_safe_get_embedding)
        return df

    @staticmethod
    def _batched(iterable, n):
        """Batch data into tuples of length n. The last batch may be shorter."""
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch

    @staticmethod
    def _chunked_tokens(text, encoding_name, chunk_length):
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        chunks_iterator = CodeProcessor._batched(tokens, chunk_length)
        yield from chunks_iterator
