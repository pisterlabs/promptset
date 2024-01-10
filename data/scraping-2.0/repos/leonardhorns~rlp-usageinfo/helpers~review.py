from __future__ import annotations
import itertools
import random
from copy import copy, deepcopy
from datetime import datetime, timezone
from typing import Iterable, Optional, Union, TYPE_CHECKING

import json
from pathlib import Path


import dateutil.parser

import helpers.label_selection as ls
from evaluation.scoring import DEFAULT_METRICS

if TYPE_CHECKING:
    from helpers.label_selection import LabelSelectionStrategyInterface


class Review:
    review_attributes = {
        "customer_id",
        "helpful_votes",
        "labels",
        "marketplace",
        "product_category",
        "product_id",
        "product_parent",
        "product_title",
        "review_body",
        "review_date",
        "review_headline",
        "star_rating",
        "total_votes",
        "verified_purchase",
        "vine",
    }

    label_attributes = {
        "createdAt",
        "datasets",
        "metadata",
        "scores",
        "usageOptions",
        "augmentations",
    }

    def __init__(self, review_id: str, data: dict) -> None:
        self.review_id = review_id
        self.data = data
        self.tokenized_datapoints = None

    def __getitem__(self, key: str) -> Union[str, int, dict]:
        if key in self.data:
            return self.data[key]

        raise ValueError(f"review '{self.review_id}' does not contain key '{key}'")

    def __eq__(self, other) -> bool:
        if isinstance(other, Review):
            return self.__key() == other.__key()
        else:
            return False

    def __key(self) -> str:
        return self.review_id

    def __hash__(self) -> int:
        return hash(self.__key())

    def __or__(self, other: "Review") -> "Review":
        return self.merge_labels(other, inplace=False)

    def __ior__(self, other: "Review") -> None:
        self.merge_labels(other, inplace=True)
        return self

    def __copy__(self):
        return Review(self.review_id, copy(self.data))

    def __deepcopy__(self, memo):
        return Review(self.review_id, deepcopy(self.data, memo))

    def __str__(self):
        simple_data = {}
        simple_data["product_title"] = self.data["product_title"]
        simple_data["review_headline"] = self.data["review_headline"]
        simple_data["review_body"] = self.data["review_body"]
        simple_data["labels"] = self.data["labels"].copy()
        for label_id, label in simple_data["labels"].items():
            simple_data["labels"][label_id] = label["usageOptions"]

        return f"Review {self.review_id} " + json.dumps(simple_data, indent=4)

    def __repr__(self):
        return json.dumps(self.data, indent=4)

    def get_labels(self) -> dict:
        return self.data.get("labels", {})

    def get_label_ids(self) -> set[str]:
        return set(self.get_labels().keys())

    def get_usage_options(self, label_id: str) -> list[str]:
        labels = self.get_label_for_id(label_id)
        return labels.get("usageOptions", []) if labels is not None else []

    def _check_strategy(self, strategy: LabelSelectionStrategyInterface) -> None:
        if not isinstance(strategy, ls.LabelSelectionStrategyInterface):
            raise ValueError(
                f"strategy '{type(strategy)}' doesn't implement LabelSelectionStrategyInterface"
            )

    def get_label_from_strategy(
        self, strategy: LabelSelectionStrategyInterface
    ) -> Optional[dict]:
        self._check_strategy(strategy)
        return strategy.retrieve_label(self)

    def get_label_id_from_strategy(
        self, strategy: LabelSelectionStrategyInterface
    ) -> Optional[str]:
        self._check_strategy(strategy)
        return strategy.retrieve_label_id(self)

    def get_labels_from_strategy(
        self, strategy: LabelSelectionStrategyInterface
    ) -> list[dict]:
        self._check_strategy(strategy)
        return strategy.retrieve_labels(self)

    def get_label_ids_from_strategy(
        self, strategy: LabelSelectionStrategyInterface
    ) -> list[str]:
        self._check_strategy(strategy)
        return strategy.retrieve_label_ids(self)

    def get_label_for_dataset(
        self, *dataset_names: Union[str, tuple[str, str]]
    ) -> Optional[dict]:
        return self.get_label_from_strategy(ls.DatasetSelectionStrategy(*dataset_names))

    def get_label_for_id(self, *label_ids: str) -> Optional[dict]:
        return self.get_label_from_strategy(ls.LabelIDSelectionStrategy(*label_ids))

    def reset_scores(self):
        for label_id, label in self.get_labels().items():
            label["scores"] = {}

    def add_label(
        self,
        label_id: str,
        usage_options: list[str],
        datasets: list[str] = [],
        metadata: dict = {},
        overwrite: bool = False,
    ) -> None:
        if not overwrite:
            assert (
                label_id not in self.get_labels()
            ), f"label '{label_id}' already exists in review '{self.review_id}'"

        self.data["labels"][label_id] = {
            # using ISO 8601 with UTC timezone, https://stackoverflow.com/a/63731605
            "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "usageOptions": usage_options,
            "scores": {},
            "datasets": datasets,
            "metadata": metadata,
            "augmentations": {},
        }

    def get_label_type(
        self, label_selection_strategy: LabelSelectionStrategyInterface
    ) -> str:
        labels = self.get_labels_from_strategy(label_selection_strategy)
        label_type = None
        for label in labels:
            if label_type == "both":
                return "both"
            else:
                if len(label["usageOptions"]) == 0:
                    if label_type == "usage":
                        label_type = "both"
                    else:
                        label_type = "no_usage"
                elif len(label["usageOptions"]) > 0:
                    if label_type == "no_usage":
                        label_type = "both"
                    else:
                        label_type = "usage"
        return label_type

    def label_has_usage_options(
        self, label_selection_strategy: LabelSelectionStrategyInterface
    ) -> bool:
        label = self.get_label_from_strategy(label_selection_strategy)
        if not label:
            raise ValueError(
                f"Review {self.review_id} does not have any matching labels"
            )
        return len(label["usageOptions"]) > 0

    def tokenize(
        self,
        tokenizer,
        text: str,
        for_training: bool,
        is_input: bool,
        max_length: int = float("inf"),
    ) -> Optional[dict]:
        from training.utils import MAX_OUTPUT_LENGTH

        max_length = (
            min(MAX_OUTPUT_LENGTH, max_length)
            if (not is_input and for_training)
            else max_length
        )

        if not is_input and for_training and len(text) == 0:
            text = "no usage options"

        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=not for_training,
            max_length=max_length,
            padding="max_length",
        )
        # Remove batch dimension, since we only have one example
        tokens["input_ids"] = tokens["input_ids"][0]
        tokens["attention_mask"] = tokens["attention_mask"][0]

        if not is_input and for_training:
            ids = tokens["input_ids"]
            # You need to set the pad tokens for the input to -100 for some Transformers (https://github.com/huggingface/transformers/issues/9770)>
            tokens["input_ids"][ids[:] == tokenizer.pad_token_id] = -100

        return tokens if len(tokens["input_ids"]) <= max_length else None

    def _get_output_texts_from_strategy(
        self, usage_options: list[str], strategy: str = None
    ) -> list[str]:
        if not strategy or strategy == "default":
            return ["; ".join(usage_options)]
        elif strategy == "flat":
            return usage_options or [""]
        elif strategy.startswith("shuffle"):
            usage_options = copy(usage_options)
            random.shuffle(usage_options)
            if not strategy.startswith("shuffle-"):
                return ["; ".join(usage_options)]

            permutation_limit = strategy.split("-")[1]
            if permutation_limit == "all":
                permutation_limit = None
            else:
                try:
                    permutation_limit = int(permutation_limit)
                    if permutation_limit < 1:
                        raise ValueError(
                            "Number of permutations must be greater than 0"
                        )
                except ValueError as e:
                    if str(e) == "Number of permutations must be greater than 0":
                        raise e
                    raise ValueError(
                        f"Could not parse number of permutations for shuffle strategy '{strategy}'",
                        "Please use 'shuffle-<number_of_permutations>' or 'shuffle-all'",
                    )

            permutations = [
                "; ".join(permutation)
                for permutation in itertools.islice(
                    itertools.permutations(usage_options), permutation_limit
                )
            ]
            return permutations
        else:
            raise ValueError(f"strategy '{strategy}' not supported")

    def get_prompt(self, prompt_id="avetis_v1") -> str:
        from langchain import PromptTemplate

        path = Path(__file__).parent.parent / "prompts.json"

        with open(path) as f:
            prompts = json.load(f)

        prompt_text = prompts["model-training"][prompt_id]["prompt"]
        prompt_input_variables = prompts["model-training"][prompt_id]["input_variables"]

        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=prompt_input_variables,
            validate_template=False,
        )

        prompt = prompt.format(
            **{
                key: value
                for key, value in self.data.items()
                if key in prompt_input_variables
            }
        )

        return prompt

    def get_tokenized_datapoints(
        self,
        selection_strategy: Optional[ls.LabelSelectionStrategyInterface] = None,
        multiple_usage_options_strategy: Optional[str] = None,
        prompt_id: str = "avetis_v1",
        **tokenization_kwargs,
    ) -> Iterable[dict]:
        def format_dict(model_input, output, review_id, source_id) -> dict:
            return {
                "input": model_input,
                "output": output,
                "review_id": review_id,
                "source_id": source_id,
            }

        model_input = self.get_prompt(prompt_id=prompt_id)
        model_input = self.tokenize(
            text=model_input,
            is_input=True,
            **tokenization_kwargs,
        )

        # Returns 0 if when no selection strategy. We are using 0 instead of None because of the dataloader
        label = (
            self.get_label_from_strategy(selection_strategy)
            if selection_strategy
            else None
        )
        if not label:
            yield format_dict(model_input, 0, self.review_id, "no_label")
            return
        output_texts = self._get_output_texts_from_strategy(
            label["usageOptions"], strategy=multiple_usage_options_strategy
        )

        for id, output_text in enumerate(output_texts):
            yield format_dict(
                model_input,
                self.tokenize(
                    text=output_text.lower(),  # we enforce lower case because model does not need to learn casing
                    is_input=False,
                    **tokenization_kwargs,
                ),
                self.review_id,
                f"{multiple_usage_options_strategy}_{id}",
            )

    def remove_label(self, label_id: str, inplace=True) -> Optional["Review"]:
        review_without_label = (
            self if inplace else Review(self.review_id, deepcopy(self.data))
        )
        review_without_label.data["labels"].pop(label_id, None)

        # remove score references to deleted labels
        for label in review_without_label.data["labels"].values():
            label["scores"].pop(label_id, None)

        if not inplace:
            return review_without_label

    def score(
        self,
        label_id: str,
        reference_label_id: str,
        metric_ids: Iterable[str] = DEFAULT_METRICS,
    ):
        from evaluation.scoring.metrics import SingleReviewMetrics

        scores = self.get_label_for_id(label_id)["scores"]

        if reference_label_id not in scores:
            scores[reference_label_id] = {}

        available_metrics = scores.get(reference_label_id, {})
        missing_metric_ids = set(metric_ids).difference(set(available_metrics.keys()))

        if len(missing_metric_ids) > 0:
            # calculate missing metrics
            from evaluation.scoring.metrics import SingleReviewMetrics

            result = SingleReviewMetrics.from_labels(
                self.get_labels(), label_id, reference_label_id
            ).calculate(missing_metric_ids, include_pos_neg_info=True)

            for metric_id, metric_tuple in result.items():
                scores[reference_label_id][metric_id] = metric_tuple

        return copy(scores[reference_label_id])

    def get_scores(
        self,
        label_id: Union[str, ls.LabelSelectionStrategyInterface],
        *reference_label_candidates: Union[str, ls.LabelSelectionStrategyInterface],
        metric_ids: Iterable[str] = DEFAULT_METRICS,
    ) -> Optional[dict]:
        if isinstance(label_id, ls.LabelSelectionStrategyInterface):
            label_id = self.get_label_id_from_strategy(label_id)

        for reference_label_candidate in copy(reference_label_candidates):
            if isinstance(
                reference_label_candidate, ls.LabelSelectionStrategyInterface
            ):
                strategy_candidates = self.get_label_ids_from_strategy(
                    reference_label_candidate
                )
                reference_label_candidates.remove(reference_label_candidate)
                reference_label_candidates += strategy_candidates

        """return scores for a specified reference label or the best scores if multiple reference labels are specified"""
        reference_label_candidates = set(reference_label_candidates).intersection(
            self.get_label_ids()
        )
        if label_id not in self.get_label_ids() or not reference_label_candidates:
            return None

        reference_scores = {}
        for reference_label_id in reference_label_candidates:
            reference_scores[reference_label_id] = self.score(
                label_id, reference_label_id, metric_ids
            )

        max_score = {m: (-1, None, None) for m in metric_ids}
        for m_id in metric_ids:
            for ref_id in reference_label_candidates:
                # reference_scores[ref_id][m_id] is a tuple (score, prediction_is_positive_usage, reference_is_positive_usage)
                score = reference_scores[ref_id][m_id][0]
                if score >= max_score[m_id][0]:
                    max_score[m_id] = reference_scores[ref_id][m_id]

        return max_score

    def merge_labels(
        self, other_review: "Review", inplace: bool = False
    ) -> Optional["Review"]:
        """Merge labels from another review into this one.

        This method is used to merge labels of the same review into this object.
        """
        assert self == other_review, "cannot merge labels of different reviews"
        existing_labels = self.get_labels()
        if not inplace:
            existing_labels = deepcopy(existing_labels)
        additional_labels = deepcopy(other_review.get_labels())

        for label_id, other_label in additional_labels.items():
            if label_id not in existing_labels:
                existing_labels[label_id] = other_label
            else:
                own_label = existing_labels[label_id]
                # validate same usage options
                assert (
                    own_label["usageOptions"] == other_label["usageOptions"]
                ), f"'{label_id}' in review '{other_review.review_id}' has inconsistent usage options"

                # merge scores
                for ref_id, other_score_dict in other_label["scores"].items():
                    if ref_id not in own_label["scores"]:
                        own_label["scores"][ref_id] = other_score_dict
                    else:
                        # merge different score metrics for same reference
                        own_label["scores"][ref_id].update(other_score_dict)

                # merge datasets and metadata
                own_label["datasets"] = list(
                    set(own_label["datasets"]) | set(other_label["datasets"])
                )
                own_label["augmentations"] |= other_label["augmentations"]
                own_label["metadata"] |= other_label["metadata"]

        if not inplace:
            return Review(
                self.review_id, self.data.copy() | {"labels": existing_labels}
            )

        self.data["labels"] = existing_labels

    def validate(self) -> None:
        error_msg_prefix = f"encountered error in review '{self.review_id}':"

        data_keys_set = set(self.data.keys())
        if not set(self.review_attributes).issubset(set(data_keys_set)):
            raise ValueError(
                f"{error_msg_prefix} wrong attribute names\n"
                f"got: {data_keys_set}\nexpected: {self.review_attributes}"
            )

        labels = self.get_labels()
        if not isinstance(labels, dict):
            raise ValueError(
                f"{error_msg_prefix} 'labels' is not of type dict but {type(labels)}"
            )

        for label_id, label in labels.items():
            if not isinstance(label, dict):
                raise ValueError(
                    f"{error_msg_prefix} label '{label_id}' is not of type dict but {type(label)}"
                )
            label_keys_set = set(label.keys())
            if label_keys_set != self.label_attributes:
                raise ValueError(
                    f"{error_msg_prefix} wrong keys in label '{label_id}'\n"
                    f"got: {label_keys_set}\nexpected: {self.label_attributes}",
                )
            if not isinstance(label["usageOptions"], list):
                raise ValueError(
                    f"{error_msg_prefix} 'usageOptions' in label '{label_id}' is not of type list but {type(label['usageOptions'])}",
                )
            if not isinstance(label["metadata"], dict):
                raise ValueError(
                    f"{error_msg_prefix} 'metadata' in label '{label_id}' is not of type dict but {type(label['metadata'])}",
                )
            if not isinstance(label["scores"], dict):
                raise ValueError(
                    f"{error_msg_prefix} 'scores' in label '{label_id}' is not of type dict but {type(label['scores'])}",
                )
            if not isinstance(label["datasets"], list):
                raise ValueError(
                    f"{error_msg_prefix} 'datasets' in label '{label_id}' is not of type dict but {type(label['datasets'])}",
                )
            try:
                dateutil.parser.isoparse(label["createdAt"])
            except Exception:
                raise ValueError(
                    f"{error_msg_prefix} 'createdAt' timestamp in label '{label_id}' is not ISO 8601",
                )
