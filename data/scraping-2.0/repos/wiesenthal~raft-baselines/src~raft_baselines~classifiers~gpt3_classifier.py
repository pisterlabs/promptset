from typing import Dict, Optional, List, Mapping

import numpy as np
import datasets
from sentence_transformers import util
import torch

from raft_baselines.classifiers.in_context_classifier import InContextClassifier
from raft_baselines.utils.gpt3_utils import (
    complete,
)
from raft_baselines.utils.tokenizers import TransformersTokenizer
from raft_baselines.utils.embedders import OpenAIEmbedder, SentenceTransformersEmbedder

GPT3_MAX_TOKENS = 2048
tokenizer = TransformersTokenizer("gpt2")


class GPT3Classifier(InContextClassifier):
    def __init__(
        self,
        *args,
        model: str = "ada",
        similarity_embedder_type: str = "openai", # | "sentence_transformers",
        **kwargs,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: str = model
        if similarity_embedder_type == "sentence_transformers":
            self.similarity_embedder = SentenceTransformersEmbedder()
        elif similarity_embedder_type == "openai":
            self.similarity_embedder = OpenAIEmbedder(max_tokens=GPT3_MAX_TOKENS)
        super().__init__(
            *args,
            tokenizer=tokenizer,
            max_tokens=GPT3_MAX_TOKENS,
            **kwargs,
        )

    def semantically_select_training_examples(
        self, target: Mapping[str, str]
    ) -> datasets.Dataset:
        formatted_examples_without_labels = tuple(
            self.format_dict(
                {col: row[col] for col in self.input_cols if col in row},
            )
            for row in self.training_data
        )
        formatted_target = self.format_dict(target)

        # adapted from https://towardsdatascience.com/semantic-similarity-using-transformers-8f3cb5bf66d6
        target_embedding = self.similarity_embedder(tuple([formatted_target]))
        example_embeddings = self.similarity_embedder(formatted_examples_without_labels)
        
        similarity_scores = util.pytorch_cos_sim(target_embedding, example_embeddings)[
            0
        ]
        
        sorted_indices = torch.argsort(-similarity_scores.to(self.device))
        return self.training_data.select(
            list(reversed(sorted_indices[: self.num_prompt_training_examples]))
        )

    def does_token_match_class(self, token: str, clas: str) -> bool:
        # prepend a space to the class label
        # because we always expect a leading space in the first token
        # returned from the OpenAI API, given our prompt format
        clas_str = (
            f" {clas}" if not self.add_prefixes else f" {self.classes.index(clas) + 1}"
        )

        clas_first_token_id: int = self.tokenizer(clas_str)["input_ids"][0]
        token_id: int = self.tokenizer(token)["input_ids"][0]

        # Compare token ids rather than the raw tokens
        # because GPT2TokenizerFast represents some special characters
        # differently from the GPT-3 API
        # (e.g. the space at the beginning of the token is " " according to the API,
        # but "Ġ" according to the tokenizer.
        # Standardizing to token ids is one easy way to smooth over that difference.
        return clas_first_token_id == token_id

    def _get_raw_probabilities(
        self,
        prompt: str,
    ) -> List[float]:
        response = complete(
            prompt,
            temperature=0.0,
            model=self.model,
            max_tokens=1,
        )
        logprobs: Dict[str, float] = response.choices[0].logprobs.top_logprobs[
            0
        ]

        raw_p = []
        for clas in self.classes:
            p = 0.0
            for token in logprobs.keys():
                if self.does_token_match_class(token, clas):
                    p += np.exp(logprobs[token])
            raw_p.append(p)

        return raw_p
