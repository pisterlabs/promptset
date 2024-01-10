import copy
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from inspect import isabstract, signature
import math
from typing import List, Dict, Tuple
import numpy as np
import random
import re
import torch

from fairseq.models import BaseFairseqModel
from fairseq import utils
import fairseq.distributed.utils as dist
from fairseq.utils import print_r0

from examples.few_shot import tasks, templates
from examples.few_shot.tasks import FewShotSample
from examples.few_shot.models import (
    convert_max_positions_to_int,
    run_with_adaptative_max_tokens,
)
import time

from examples.few_shot.openai_api_utils import call_openai_completion, openai_result_to_fairseq_result, openai_result_to_json
from examples.few_shot.huggingface_api_utils import HuggingFaceAPIHelper

SCORES = "positional_scores"

PREDICTORS_REGISTRY = {}
CALIBRATORS_REGISTRY = {}


def get_calibrator_class_by_name(calibrator_name):
    return CALIBRATORS_REGISTRY[calibrator_name.lower()]


def get_predictor_class_by_name(predictor_name):
    return PREDICTORS_REGISTRY[predictor_name.lower()]


def get_all_predictors():
    return list(PREDICTORS_REGISTRY.keys())

def calc_ppl(positional_scores: torch.Tensor):
    return float(positional_scores.mean().neg().exp())

class ScoredCandidate:
    def __init__(self, candidate, score: float, meta=None):
        self.candidate = candidate
        self.score = score
        self.meta = meta


class Prediction:
    def __init__(self, sample, scored_candidates, meta=None):
        self.sample = sample
        self.scored_candidates = scored_candidates
        self.meta = meta

    @property
    def best_candidate(self):
        return max(self.scored_candidates, key=lambda x: x.score)


@dataclass
class FewShotCalibrator(ABC):
    @abstractmethod
    def calibrate_predictions(
        self,
        samples: List[FewShotSample],
        predictions: List[Prediction],
        calibration_samples: List[FewShotSample],
        calibration_predictions: List[Prediction],
    ):
        pass

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Allows instanciation from a kwargs dict even if it contains unused keys"""
        return cls(
            **{k: v for k, v in kwargs.items() if k in signature(cls).parameters}
        )

    @classmethod
    def get_calibrator_name(cls):
        [name] = re.match("(.+)Calibrator", cls.__name__).groups()
        return name.lower()

    def __init_subclass__(cls, **kwargs):
        """Register all children in registry"""
        super().__init_subclass__(**kwargs)
        if isabstract(cls):
            return
        calibrator_name = cls.get_calibrator_name()
        assert (
            calibrator_name not in CALIBRATORS_REGISTRY
        ), f"{calibrator_name} calibrator already registered!"
        CALIBRATORS_REGISTRY[calibrator_name] = cls


@dataclass
class AverageOptionCalibrator(FewShotCalibrator):
    def get_candidate_scores(
        self,
        scored_candidates: List[ScoredCandidate],
    ) -> List[Tuple[str, float]]:
        """
        This converts the scored candidates to tuple with the candidate label and the score converted to float (instead of tensor).
        This method also can be used for normalization such as softmax, etc. 
        """
        processed_candidates = [(x.candidate, float(x.score)) for x in scored_candidates]
        return processed_candidates


    def aggregate_calibration_predictions(
        self, calibration_sample, sample2prediction
    ) -> Dict[str, float]:
        """
        Here we aggregate the predictions from calibration candidates in case we have more than one calibration option.
        """
        sample_subproblems = (
            calibration_sample.subproblems
            if calibration_sample.has_subproblems
            else [calibration_sample]
        )

        per_sample_predictions = []
        per_cand_metas = {}
        for subproblem in sample_subproblems:
            prediction = sample2prediction[subproblem]
            
            # get scores
            scored_candidates = self.get_candidate_scores(
                prediction.scored_candidates
            )
            per_sample_predictions.append(scored_candidates)

            # get meta
            for cand in prediction.scored_candidates:
                if cand.candidate not in per_cand_metas:
                    per_cand_metas[cand.candidate] = []
                per_cand_metas[cand.candidate].append(cand.meta)
            

        aggregated_predictions = get_average_predictions(per_sample_predictions)

        return aggregated_predictions, per_cand_metas

    @classmethod
    def get_calibrator_name(cls):
        return "average_option"

    def calibrate_scored_candidates(
        self,
        scored_candidates: List[ScoredCandidate],
        aggregated_calibration_predictions: Dict[str, float],
    ):
        """This calibrartes the scored candidates using the aggragated calibration predicitons.

        Args:
            scored_candidates (List[ScoredCandidate]): Scored sample candidates.
            aggregated_calibration_predictions (Dict[str, float]): Aggregated calibration scores per candidate. 
        """
        
        # update values
        for scored_candidate in scored_candidates:
            cand_name = scored_candidate.candidate
            cand_score = float(scored_candidate.score) # get float from tensor
            
            calibration_score = aggregated_calibration_predictions[cand_name]

            calibrated_cand_score = (
                cand_score - calibration_score  # these are logprobs
            )
            scored_candidate.score = calibrated_cand_score
            scored_candidate.meta["calib_score"] = calibration_score


    def calibrate_predictions(
        self,
        samples: List[FewShotSample],
        predictions: List[Prediction],
        calibration_samples: List[FewShotSample],
        calibration_predictions: List[Prediction],
    ):
        """
        Calibrates the score_candidates in predictions!
        For each sample, we aggregate the calibration_predictions for the corresponding calibration_sample
        and calibrate the sample (or its subproblems) predictions with the this score.

        Example: NLI with single (premise, hypothesis, entailement/not_entailment label) can have multiple such calibration records.
        We calculate the entailment  vs not_entailment label scores from the calibration records,
        average them and use them for calibration of the original example.
        """
        assert len(samples) == len(calibration_samples)

        sample2prediction = {
            prediction.sample: prediction for prediction in predictions
        }
        calib_sample2prediction = {
            prediction.sample: prediction for prediction in calibration_predictions
        }

        for sample, calibration_sample in zip(samples, calibration_samples):
            candidate_calibration_scores, per_cand_metas = self.aggregate_calibration_predictions(
                calibration_sample, calib_sample2prediction
            )
            sample_subproblems = (
                sample.subproblems if sample.has_subproblems else [sample]
            )

            for subproblem in sample_subproblems:
                prediction = sample2prediction[subproblem]
                self.calibrate_scored_candidates(
                    prediction.scored_candidates, candidate_calibration_scores
                )
                for scored_cand in prediction.scored_candidates:
                    scored_cand.meta["calib_metas"] = per_cand_metas.get(scored_cand.candidate)


def get_average_predictions(
    dummy_option_predictions: List[Tuple[str, float]]
) -> Dict[str, float]:
    dummy_predictions_avg = {}
    for pred in dummy_option_predictions:
        for cand, score in pred:
            if cand not in dummy_predictions_avg:
                dummy_predictions_avg[cand] = [score]
            else:
                dummy_predictions_avg[cand].append(score)

    # calibrate with dummy predictions
    dummy_predictions_avg = {
        k: sum(v) / len(v) for k, v in dummy_predictions_avg.items()
    }

    return dummy_predictions_avg


class FewShotPredictor(ABC):

    def __init__(self, *, model: BaseFairseqModel, task: tasks.FewShotTask, replace_newline_with_eos=False, **kwargs):
        self.model = model
        self.task = task
        self._langs_mapping = None
        self.replace_newline_with_eos = replace_newline_with_eos
        self.batch_size = kwargs.get("batch_size", 1)

    @abstractmethod
    def predict(self, samples):
        pass

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Allows instanciation from a kwargs dict even if it contains unused keys"""
        return cls(**kwargs)

    @classmethod
    def get_predictor_name(cls):
        [name] = re.match("(.+)Predictor", cls.__name__).groups()
        return name.lower()

    def __init_subclass__(cls, **kwargs):
        """Register all children in registry"""
        super().__init_subclass__(**kwargs)
        if isabstract(cls):
            return
        predictor_name = cls.get_predictor_name()
        assert (
            predictor_name not in PREDICTORS_REGISTRY
        ), f"{predictor_name} predictor already registered!"
        PREDICTORS_REGISTRY[predictor_name] = cls

    @property
    def langs(self):
        return self.model.langs

    @property
    def add_lang_bos_token(self):
        return self.model.add_lang_bos_token

    @property
    def langs_mapping(self):
        if self._langs_mapping is None and self.langs is not None:
            self._langs_mapping = {}
            langs_list = list(sorted([x.strip() for x in self.langs.split(",")]))
            for lang in langs_list:
                lang_base = lang.split("_")[0]
                # given en_EN, add both en and en_EN to the mapping
                for lang_var in [lang_base, lang]:
                    if lang_var not in self._langs_mapping:
                        self._langs_mapping[lang_var] = lang

        return self._langs_mapping

    def get_lang_prefix(self, text_lang):
        # HACK: add lang code to prompt for models using language token
        # See get_sentence_and_language in fairseq/hub_utils.py
        if self.langs is None or text_lang is None:
            return ""

        lang_code = self.langs_mapping.get(text_lang, None)
        if lang_code is None:
            return ""
            
        lang_token = self.model.to_lang_token(lang_code)
        return f"<lang>{lang_token}</lang>"

    def encode_batch(self, sentences):
        return [self.encode(sentence) for sentence in sentences]
    
    def decode_batch(self, sentences):
        return [self.decode(sentence) for sentence in sentences]

    def encode(self, sentence: str) -> torch.LongTensor:
        lang, sentence = self.model.get_sentence_and_language(sentence)

        sentence = self.model.tokenize(sentence)

        if self.replace_newline_with_eos:
            # We remove consecutive newlines, which indicated EOD during training
            lines = [self.model.apply_bpe(line) for line in sentence.splitlines() if line]
            sentence = " </s> ".join(lines)
        else:
            sentence = self.model.apply_bpe(sentence)

        if lang is not None:
            sentence = f"{lang} {sentence}"

        binarized_sentence = self.model.binarize(sentence)
        return binarized_sentence

    def decode(self, tokens: torch.LongTensor) -> str:
        if self.replace_newline_with_eos:
            assert len(tokens.shape) == 1
            if tokens[-1] == self.dictionary.eos():
                tokens = tokens[:-1]  # Remove final EOS
            eos_inds = (tokens == self.dictionary.eos()).nonzero(as_tuple=False).flatten().tolist()
            eos_inds.append(len(tokens))
            split_tokens = [tokens[:eos_inds[0]]]
            split_tokens += [tokens[eos_inds[i]+1:eos_inds[i+1]] for i in range(len(eos_inds)-1)]
        else:
            split_tokens = [tokens]

        sentences = [self.model.string(tokens) for tokens in split_tokens]

        # Remove the lang token
        sent_split = sentences[0].split(" ", 1)
        lang_token = None
        if sent_split[0] in self.model.lang_tokens:
            lang_token = sent_split[0]
            sentences[0] = sent_split[1]

        sentences = [self.model.remove_bpe(sentence) for sentence in sentences]
        sentences = [self.model.detokenize(sentence) for sentence in sentences]

        if lang_token is not None:
            sentences[0] = self.model.add_language_to_sentence(sentences[0], lang_token)

        return "\n".join(sentences)

    def sample_with_adaptative_max_tokens(self, sentences, beam=1, verbose=False, **kwargs):
        tokenized_sentences = self.encode_batch(sentences)
        batched_hypos = run_with_adaptative_max_tokens(
            self.model, self.model.generate,
            tokenized_sentences=tokenized_sentences, beam=beam, verbose=verbose, **kwargs
        )
        return self.decode_batch([hypos[0]["tokens"] for hypos in batched_hypos])

    @property
    def max_positions(self):
        return convert_max_positions_to_int(self.model.max_positions)

    @property
    def dictionary(self):
        return self.model.task.dictionary


class RandomPredictor(FewShotPredictor):
    def predict(self, samples):
        assert self.task.has_candidates
        samples = [
            subproblem
            for sample in samples
            for subproblem in (
                sample.subproblems if sample.has_subproblems else [sample]
            )
        ]  # Expand samples with subproblems (e.g., MultiRC)
        predictions = [
            Prediction(
                sample=sample,
                scored_candidates=[
                    ScoredCandidate(
                        candidate=np.random.choice(sample.candidates), score=1.0
                    )
                ],
            )
            for sample in samples
        ]

        return predictions, {}


class MajorityClassPredictor(FewShotPredictor):
    def predict(self, samples):
        # We are picking the majority class in the test set
        assert self.task.has_candidates
        samples = [
            subproblem
            for sample in samples
            for subproblem in (
                sample.subproblems if sample.has_subproblems else [sample]
            )
        ]  # Expand samples with subproblems (e.g., MultiRC)
        correct_candidates = [
            candidate for sample in samples for candidate in sample.correct_candidates
        ]
        majority_class = Counter(correct_candidates).most_common()[0][0]
        predictions = [
            Prediction(
                sample=sample,
                scored_candidates=[
                    ScoredCandidate(candidate=majority_class, score=1.0)
                ],
            )
            for sample in samples
        ]

        return predictions, {}


class PromptingPredictor(FewShotPredictor):

    def __init__(
        self, *,
        template: templates.FewShotTemplate,
        calibrator: FewShotCalibrator = None,
        train_sep: str = " ",
        truncate_few_shot_samples_globally = False,
        no_shuffle_exampes = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.template = template
        self.calibrator = calibrator
        self.train_sep = train_sep
        self.truncate_few_shot_samples_globally = truncate_few_shot_samples_globally
        self.shuffle_examples = not truncate_few_shot_samples_globally
        if self.shuffle_examples and self.truncate_few_shot_samples_globally:
            raise ValueError("--truncate-few-shot-samples-globally requires --no-shuffle-examples")

    @property
    def use_calibration(self):
        return self.calibrator is not None and (self.task.calibration_options is not None and len(self.task.calibration_options) > 0)
        
    def get_prompts(self, samples):

        def token_count(sentence, remove_mask=False):
            if len(sentence) == 0:
                return 0
            if remove_mask:
                return sum([token_count(s) for s in sentence.split("<mask>") if s])
            return max(1, self.encode(sentence).numel() - 1)  # Don't count </s>

        if self.shuffle_examples:
            rand = random.Random()
            rand.seed(0)  # We use a fixed seed for reproducibility
        train_samples = [self.template.encode_correct_candidate(sample) for sample in self.task.train_samples]
        sep_len = token_count(self.train_sep)
        train_sample_lens = [token_count(sample) + sep_len for sample in train_samples]
        description = self.template.task_description
        description_len = 0 if description is None else token_count(description) + sep_len

        # Step 1: compute maximum sample length without context and cache instantiated prompts and candidates
        #   Running example:
        #     - prompts_without_context: ['The bowling ball knocked over the bowling pins because <mask>']
        #     - candidates: [['the man rolled the bowling ball down the alley.', 'the man dropped the bowling ball on his foot.']]
        #     - lens_without_context: len(prompt_wo_context_wo_mask) + max_candidate_len
        prompts_without_context = []
        candidates = [] if self.task.has_candidates else None
        lens_without_context = []    # length of each test sample prompt without the few-shot context
        for sample in samples:
            prompt = self.template.encode(sample)   # test sample prompt without context
            prompts_without_context.append(prompt)
            prompt_len = token_count(prompt, remove_mask=True)
            if self.task.has_candidates:
                sample_candidates = [
                    self.template.verbalize(sample, candidate)
                    for candidate in sample.candidates
                ]
                sample_candidate_len = max([token_count(candidate) for candidate in sample_candidates])
                candidates.append(sample_candidates)
            else:
                sample_candidate_len = self.task.get_max_candidate_length()
            lens_without_context.append(prompt_len + sample_candidate_len)    
        max_len_without_context = max(lens_without_context)
                
        # Step 2: prepare model input, truncate to under max_positions when needed.
        # Running example:
        #     - train_samples: ("The group overlooked the woman's faux pas so the woman was relieved.",)
        #     - prompts_with_context: ["The group overlooked the woman's faux pas so the woman was relieved. The bowling ball knocked over the bowling pins because <mask>"]
        #     - ntrain_samples: number of training samples fit under max_positions for each example
        prompts_with_context, ntrain_samples = [], []
        max_tgt_len, n_truncated = 0, 0
        for sample, prompt_wo_context, len_wo_context in zip(samples, prompts_without_context, lens_without_context):
            if self.truncate_few_shot_samples_globally:
                len_wo_context = max_len_without_context
            if self.shuffle_examples and len(train_samples) > 0:
                aux = list(zip(train_samples, train_sample_lens))
                rand.shuffle(aux)
                train_samples, train_sample_lens = zip(*aux)
            # Reserve one position for </s>, one position for a special token
            remaining_positions = (self.max_positions - 1)  
            remaining_positions -= 1
            # Reserve one position for language token if required
            if self.langs is not None and self.add_lang_bos_token:
                lang_code = sample.data.get("lang", self.template.lang_code)
                lang_prefix = self.get_lang_prefix(lang_code)
                remaining_positions -= 1
            else:
                lang_prefix = ""
            remaining_positions -= len_wo_context
            # Truncate sample without context if it cannot fit the maximum input positions
            if remaining_positions < 0:
                s = prompt_wo_context.split("<mask>")
                for i in range(len(s)):
                    toks = self.encode(s[i])
                    if toks.numel() <= -remaining_positions:
                        remaining_positions += toks.numel()
                        s[i] = ""
                    else:
                        s[i] = self.decode(toks[-remaining_positions:])
                        remaining_positions = 0
                    if remaining_positions >= 0:
                        break
                prompt_wo_context = "<mask>".join(s)
                n_truncated += 1
            assert remaining_positions >= 0, "Sample without context is longer than max_positions after truncating"

            context = []
            if description is not None and remaining_positions >= description_len:
                context.append(description)
                remaining_positions -= description_len
            ntrain_samples.append(0)
            kept_train_sample_lens = []
            for i, (train_sample, train_sample_len) in enumerate(zip(train_samples, train_sample_lens)):
                kept_train_sample_lens.append(train_sample_len)
                if train_sample_len > remaining_positions:
                    break
                else:
                    remaining_positions -= train_sample_len
                    context.append(train_sample)
                    ntrain_samples[-1] += 1
            assert remaining_positions >= 0, "Sample with context is longer than max_positions"
            prompts_with_context.append(lang_prefix + self.train_sep.join(context + [prompt_wo_context]))
            max_tgt_len = max(max_tgt_len, self.max_positions-remaining_positions)

        print_r0(f"expected_max_tgt_len={max_tgt_len}, max_positions={self.max_positions}")
        if n_truncated > 0:
            print_r0(f"WARNING: Truncated {n_truncated} test samples that were longer than max_positions")
        
        return prompts_with_context, candidates, ntrain_samples


class CLMPromptingPredictor(PromptingPredictor):
    
    def __init__(self, *, beam_size: int = 1, scoring: str = "sum", 
                 add_prompt_to_meta: bool = False, 
                 add_positional_scores_to_meta: bool = False, 
                 add_prompt_tokens_to_meta: bool = False, 
                 add_calib_to_meta: bool = False,
                 compute_vocab_dist: bool = False, 
                 **kwargs):
        super().__init__(**kwargs)
        self.beam_size = beam_size
        self.scoring = scoring
        self.add_prompt_to_meta = add_prompt_to_meta
        self.add_positional_scores_to_meta = add_positional_scores_to_meta
        self.add_prompt_tokens_to_meta = add_prompt_tokens_to_meta
        self.add_calib_to_meta = add_calib_to_meta
        self.compute_vocab_dist = compute_vocab_dist

    def get_candidate_score(self, hypothesis, unconditional_hypothesis, common_prefix_length, common_suffix_length):
        if self.scoring == "mean":
            score = hypothesis[SCORES][common_prefix_length:].mean()
        elif self.scoring == "unconditional-norm":
            score = hypothesis[SCORES][common_prefix_length:].sum()
            candidate_len = hypothesis['tokens'].numel() - common_prefix_length
            unconditional_score = unconditional_hypothesis[SCORES][-candidate_len:].sum()
            assert hypothesis['tokens'][common_prefix_length:].equal(unconditional_hypothesis['tokens'][-candidate_len:])
            score -= unconditional_score
        elif self.scoring == "sum":
            score = hypothesis[SCORES].sum()
        elif self.scoring == "suffix":
            score = hypothesis[SCORES][-common_suffix_length:].sum()
        else:
            raise NotImplementedError(f"Unknown scoring {self.scoring}")
        return score

    @staticmethod
    def gather_hypotheses(local_hypotheses, n_examples, max_tgt_len) -> List[Dict]:
        """Gather each workers local SCORES into one dictionary."""
        pscores_buffer = torch.zeros((n_examples, max_tgt_len), dtype=torch.float, device=torch.cuda.current_device())
        ntok_buffer = torch.zeros(n_examples, dtype=torch.int, device=torch.cuda.current_device())
        for h in local_hypotheses:
            n_tok = len(h[SCORES])
            ntok_buffer[h['id']] = n_tok
            pscores_buffer[h['id'], :n_tok] = h[SCORES].clone()  # NOTE(SS): need clone?
        torch.distributed.all_reduce(pscores_buffer)
        torch.distributed.all_reduce(ntok_buffer)
        assert not pscores_buffer.eq(0).all(1).any(), 'some ids are not present'
        # Remove trailing 0s in each prediction to allow mean(score) type statistics
        scores = [pscores_buffer[i][:ntok_buffer[i]] for i in range(n_examples)]
        return [{SCORES: s} for s in scores]


    @staticmethod
    def get_common_prefix_and_suffix_lengths(tokens_list):
        # This method supports both results from openai api call which returns list of of str tokens 
        # and the results from fairseq models which returns torch.Tensor

        if isinstance(tokens_list[0], torch.Tensor):
            tokens_list = [x.tolist() for x in tokens_list]
            
        tokens_list = [np.array(x) for x in tokens_list]
        n = min([len(tokens) for tokens in tokens_list])
        # Compute common prefix length
        prefix_len = n
        first = tokens_list[0][:n]
        for tokens in tokens_list[1:]:
            neq_inds = (first != tokens[:n]).nonzero()[0]
            if len(neq_inds) > 0:
                prefix_len = min(prefix_len, neq_inds[0].item())

        # Compute common suffix length
        suffix_len = n
        first = tokens_list[0][-n:]
        for tokens in tokens_list[1:]:
            neq_inds = (first != tokens[-n:]).nonzero()[0]
            if len(neq_inds) > 0:
                suffix_len = min(suffix_len, n - 1 - neq_inds[-1].item())

        return prefix_len, suffix_len

    def score_hypotheses(self, samples, hypotheses_with_prompts):
        predictions = []
        for sample in samples:
            hypotheses_batch, prompts_batch = zip(*[hypotheses_with_prompts.pop(0) for _ in sample.candidates])

            if self.scoring == "unconditional-norm":
                unconditional_hypotheses_batch = [hypotheses_with_prompts.pop(0)[0] for _ in sample.candidates]
            else:
                unconditional_hypotheses_batch = [None for _ in sample.candidates]
            if getattr(self.task, "single_token_mlm_eval", False):
                # Only need predictions at single position; also assuming one token prediction
                prefix_len, suffix_len = self.get_common_prefix_and_suffix_lengths([hypotheses_batch[0]['tokens'], self.encode(self.template.encode(sample).replace("<mask>", "UNK"))[:-1]])
            else:
                prefix_len, suffix_len = self.get_common_prefix_and_suffix_lengths([hypo["tokens"] for hypo in hypotheses_batch])
            
            scored_candidates = []
            common_prefix_ppl = None

            for candidate, hypothesis, unconditional_hypothesis, prompt in zip(sample.candidates, hypotheses_batch, unconditional_hypotheses_batch, prompts_batch):
                if common_prefix_ppl is None:
                    common_prefix_score = hypothesis[SCORES][:prefix_len].mean()
                    common_prefix_ppl = float(common_prefix_score.neg().exp())

                cand_score = self.get_candidate_score(hypothesis, unconditional_hypothesis, prefix_len, suffix_len)
                cand_ppl = calc_ppl(hypothesis[SCORES][prefix_len:])
                cand_full_sequence_ppl = calc_ppl(hypothesis[SCORES])

                if len(hypothesis[SCORES]) == prefix_len:  # Debug weird PPL values
                    print_r0(f'Warning: same prefix len as n positional scores: lens: ({prefix_len}, {suffix_len}), cand_score: {cand_score}, cand_ppl: {cand_ppl}, scores_len: {len(hypothesis[SCORES])}')

                scored_candidate_meta = {"ppl": cand_ppl, "ppl_full": cand_full_sequence_ppl, "score_raw": float(cand_score)}
                if self.add_prompt_to_meta:
                    scored_candidate_meta["prompt"] = prompt

                if self.add_positional_scores_to_meta:
                    scored_candidate_meta[SCORES] = [float(x) for x in list(hypothesis[SCORES])]

                if self.add_prompt_tokens_to_meta:
                    scored_candidate_meta["prompt_tokens"] = [x.item() if isinstance(x, torch.Tensor) else x for x in list(hypothesis["tokens"])]
                
                # Only need predictions at single position; Retrieve topk candidates and correct candidate's rank
                if getattr(self.task, "single_token_mlm_eval", False):
                    assert hypothesis["vocab_dist"] is not None, "Use `--compute-vocab-dist` argument to return vocab distribution."
                    sorted_candidates_indices =  torch.sort(hypothesis["vocab_dist"][prefix_len], descending=True).indices
                    topk_candidates = sorted_candidates_indices[:self.beam_size]
                    scored_candidate_meta["top_ranked_predictions"] = [self.decode([cand_tokens]).strip() for cand_tokens in topk_candidates.tolist()]
                    scored_candidate_meta["rank"] = sorted_candidates_indices.tolist().index(hypothesis["tokens"][prefix_len]) + 1

                scored_candidates.append(
                    ScoredCandidate(
                        candidate=candidate, score=cand_score, meta=scored_candidate_meta
                    )
                )
            predictions.append(
                Prediction(
                    sample=sample,
                    scored_candidates=scored_candidates,
                    meta={"ppl": common_prefix_ppl},
                )
            )

        return predictions

    @torch.no_grad()
    def score_candidates(self, samples) -> List[Prediction]:
        prompts: List[str] = []
        prompts_with_mask, candidates, ntrain = self.get_prompts(samples)
        for prompt, cands in zip(prompts_with_mask, candidates):
            prompts.extend(
                [prompt.replace("<mask>", candidate) for candidate in cands]
            )
            if self.scoring == "unconditional-norm":
                prompts.extend(["Answer: " + candidate for candidate in cands])
        
        print_r0(f"Average number of train samples: {np.mean(ntrain):.2f}")

        # Predict
        unique_prompts = copy.deepcopy(prompts)
        
        print_r0(
            "Predicting {0} samples with {1} prompts..".format(
                len(samples), len(unique_prompts)
            )
        )

        tokenized_sentences = self.encode_batch(unique_prompts)
        max_seq_len = max(x.numel() for x in tokenized_sentences) + 1
        ws = dist.get_global_world_size()
        min_examples_per_worker = math.floor(len(tokenized_sentences) / ws)
        if min_examples_per_worker == 0:
            n_extra_examples = ws - len(tokenized_sentences)
            tokenized_sentences.extend(tokenized_sentences[:n_extra_examples])
            print_r0(f'Adding {n_extra_examples} dummy examples to allow min_examples_per_worker=1')
            min_examples_per_worker = math.floor(len(tokenized_sentences) / ws)
            assert min_examples_per_worker == 1
        else:
            n_extra_examples = 0
        bs = 1
        # To experiment with batch size > 1, uncomment the next line
        # bs = max(1, min(math.floor(self.model.cfg.dataset.max_tokens / max_seq_len), self.batch_size))
        print_r0(f'Before running model, bs={bs}, max_tgt_len={max_seq_len} mem={torch.cuda.max_memory_allocated()/(1024**3):.2f}GB')
        local_hypotheses = self.model.generate(
            tokenized_sentences=tokenized_sentences, 
            score_reference=True, 
            batch_size=bs, 
            compute_vocab_dist=self.compute_vocab_dist
        )
        local_hypotheses = [h[0] for h in local_hypotheses]
        if ws <= 1:
            unique_hypotheses = local_hypotheses
        else:
            unique_hypotheses = self.gather_hypotheses(local_hypotheses,  len(tokenized_sentences), max_seq_len)
            for i, x in enumerate(unique_hypotheses):
                x['tokens'] = tokenized_sentences[i][1:]  # Remove EOS for consistency with hub_utils.py
        if n_extra_examples > 0:
            unique_hypotheses = unique_hypotheses[:-n_extra_examples]
        del tokenized_sentences

        utils.assert_equal(len(unique_hypotheses), len(unique_prompts))
        prompt2hypothesis = {
            prompt: hypothesis
            for prompt, hypothesis in zip(unique_prompts, unique_hypotheses)
        }

        hypotheses_with_prompts = [(prompt2hypothesis[prompt], prompt) for prompt in prompts]

        # Score the results
        predictions = self.score_hypotheses(samples, hypotheses_with_prompts)
        for prediction, nb_trunc_few_shot_samples in zip(predictions, ntrain):
            if prediction.meta is None:
                prediction.meta = dict()
            prediction.meta["nb_trunc_few_shot_samples"] = nb_trunc_few_shot_samples

        return predictions

    def generate(self, samples):
        prompts = []
        prompts_with_mask, _, ntrain = self.get_prompts(samples)
        for prompt in prompts_with_mask:
            assert prompt.endswith("<mask>")
            prompt = prompt[:-6].rstrip()  # Remove <mask>
            prompts.append(prompt)
        
        print_r0(f"Average number of train samples: {np.mean(ntrain):.2f}")

        # Predict
        generations = self.sample_with_adaptative_max_tokens(
            sentences=prompts,
            beam=self.beam_size,
            max_len_b=self.max_positions,
        )

        # Postprocess
        assert(len(generations) == len(ntrain))
        predictions = []
        for i, (generation, prompt, sample) in enumerate(zip(generations, prompts, samples)):
            assert (
                generation[: len(prompt)] == prompt
            ), f"{generation}\n\n{prompt}"  # TODO: This does not pass for seq2seq models yet
            generation = generation[
                len(prompt) :
            ]  # Strip the prompt to keep only actual generated text
            generation = self.template.postprocess(sample, generation)
            meta = {"num_trunc_few_shot_examples": ntrain[i]}
            if self.add_prompt_to_meta:
                meta["prompt"] = prompt
            predictions.append(
                Prediction(
                    sample=sample,
                    scored_candidates=[
                        ScoredCandidate(candidate=generation, score=1.0, meta=meta)
                    ],
                )
            )
        return predictions

    def predict_outputs(self, samples):
        samples = [
            subproblem
            for sample in samples
            for subproblem in (
                sample.subproblems if sample.has_subproblems else [sample]
            )
        ]  # Expand samples with subproblems (e.g., MultiRC)
        if (
            self.task.has_candidates
        ):  # For multiple choice tasks we score all candidates
            return self.score_candidates(samples)
        else:
            return self.generate(samples)

    def predict(self, samples):
        if self.use_calibration:
            return self.predict_with_calibration(samples)
        else:
            return self.predict_without_calibration(samples)

    def get_ppl_by_candidates(self, predictions, meta_key="ppl"):
        cands_ppl = {}
        for pred in predictions:
            for sc in pred.scored_candidates:
                if sc.candidate not in cands_ppl:
                    cands_ppl[sc.candidate] = [sc.meta[meta_key]]
                else:
                    cands_ppl[sc.candidate].append(sc.meta[meta_key])

        cands_ppl = {k: np.mean(v) for k, v in cands_ppl.items()}

        return cands_ppl

    def get_ppl_for_all_candidates(self, predictions, meta_key="ppl"):
        cand_ppl = []
        for pred in predictions:
            for sc in pred.scored_candidates:
                cand_ppl.append(sc.meta[meta_key])

        cand_ppl_mean = np.mean(cand_ppl)

        return cand_ppl_mean

    def get_meta_metrics(self, eval_predictions, key="") -> dict:
        """
        Use this to extract debug info from the raw predictions' metadata.
        """
        # caluclate ppl for actual samples
        if len(key) > 0 and key[0] != "_":
            key = "_" + key

        meta_metrics = {
            f"ppl{key}_common_prefix": np.mean(
                [x.meta["ppl"] for x in eval_predictions]
            ),
            f"ppl{key}_selected_candidate": np.mean(
                [x.best_candidate.meta["ppl"] for x in eval_predictions]
            ),
            f"ppl{key}_full_selected_candidate": np.mean(
                [x.best_candidate.meta["ppl_full"] for x in eval_predictions]
            ),
        }

        meta_metrics.update({f"ppl{key}_candidates_full_prompt__{k}": v for k, v in
                             self.get_ppl_by_candidates(eval_predictions, meta_key="ppl_full").items()})
        meta_metrics[f"ppl{key}_candidates_full_prompt"] = self.get_ppl_for_all_candidates(
            eval_predictions, "ppl_full"
        )

        meta_metrics[f"ppl{key}_candidates"] = self.get_ppl_for_all_candidates(
            eval_predictions
        )

        meta_metrics["nb_trunc_few_shot_samples"] = \
            np.mean([x.meta["nb_trunc_few_shot_samples"] for x in eval_predictions])
        return meta_metrics

    def predict_without_calibration(
        self,
        samples: List[FewShotSample],
    ):
        predictions = self.predict_outputs(samples)
        try:
            meta_metrics = self.get_meta_metrics(predictions)
        except:  # The above raises an exception for generation tasks
            meta_metrics = {}
        return predictions, meta_metrics

    def predict_with_calibration(
        self,
        samples: List[FewShotSample],
    ):
        predictions, meta_metrics = self.predict_without_calibration(samples)

        print_r0("Generating calibration samples...")
        calibration_samples = self.task.build_calibration_samples_bulk(samples)
        print_r0("Predicting calibration samples...")
        calibration_predictions = self.predict_outputs(calibration_samples)
        calibration_meta_metrics = self.get_meta_metrics(
            calibration_predictions, key="calib"
        )
        meta_metrics.update(calibration_meta_metrics)

        self.calibrator.calibrate_predictions(
            samples, predictions, calibration_samples, calibration_predictions
        )

        return predictions, meta_metrics


class CLMPromptingOpenaiApiPredictor(CLMPromptingPredictor):
    enabled_engines = ["ada", "babbage", "curie",
            "ada-instruct-beta", "babbage-instruct-beta",
            "curie-instruct-beta-v2"] # add davinci models later to ensure we dont call the expensive api
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        model_name = kwargs.get("model_name_display", kwargs.get("model_name", None))
        assert model_name.startswith("openai_"), "Please use model starting with `openai_`!"
        
        self.engine = model_name.split("_")[-1]
        
        assert self.engine in self.enabled_engines, f"Engine `{self.engine}` is not supported! Supported are: {self.enabled_engines}" 

    api_call_max_tries = 70  # In case of error we have 1 sec sleep and a new call. This should prevent some minor connection loss. 
    
    def generate(self, samples):
        raise NotImplementedError()

    def score_prompts(self, prompts, 
                    max_requests_per_minute=450  # Ensure we dont hit the max per minute. Empirical experiments showed that this is 600. 
                    ):
        scored_results = []
        time_between_requests = 60.0/max_requests_per_minute
        for prompt in prompts:
            start_time = time.time()
            result = call_openai_completion(prompt, self.engine, max_tries=self.api_call_max_tries)
            
            fairseq_result = openai_result_to_fairseq_result(result)
            
            # Delay to meet the limits of reqs per minute.
            elapsed = time.time() - start_time
            delay = time_between_requests - elapsed
            if delay > 0.0:
                time.sleep(delay)  
            
            scored_results.append([fairseq_result])
            
        return scored_results

    @torch.no_grad()
    def score_candidates(self, samples) -> List[Prediction]:
        prompts: List[str] = []
        prompts_with_mask, candidates, ntrain = self.get_prompts(samples)
        for prompt, cands in zip(prompts_with_mask, candidates):
            prompts.extend(
                [prompt.replace("<mask>", candidate) for candidate in cands]
            )
        
        print_r0(f"Average number of train samples: {np.mean(ntrain):.2f}")

        # Predict
        unique_prompts = copy.deepcopy(prompts)
        
        print_r0(
            "Predicting {0} samples with {1} prompts..".format(
                len(samples), len(unique_prompts)
            )
        )

        local_hypotheses = self.score_prompts(prompts=unique_prompts)
        
        local_hypotheses = [h[0] for h in local_hypotheses]
        unique_hypotheses = local_hypotheses
        
        utils.assert_equal(len(unique_hypotheses), len(unique_prompts))
        prompt2hypothesis = {
            prompt: hypothesis
            for prompt, hypothesis in zip(unique_prompts, unique_hypotheses)
        }

        hypotheses_with_prompts = [(prompt2hypothesis[prompt], prompt) for prompt in prompts]

        # Score the results
        predictions = self.score_hypotheses(samples, hypotheses_with_prompts)
        
        return predictions
    
class CLMPromptingHuggingFacePredictor(CLMPromptingPredictor):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        model_name = kwargs.get("model_name_display", kwargs.get("model_name", None))
        assert model_name.startswith("huggingface_"), "Please use model starting with `huggingface_`!"
        
        self.model_name = model_name.split("huggingface_")[-1]
        supported_models_list = ["gpt2", "gpt2-xl", "EleutherAI=gpt-neo-2.7B", 
            "bigscience=T0_3B", "bigscience=T0pp"]
        assert self.model_name in supported_models_list, \
            f"Currently, HuggingFace API is tested only on the following models: {supported_models_list}"  
        
        # we change /s to = for directory creation, reverse for use with API
        self.model_name.replace('/', '=')
        self.helper = HuggingFaceAPIHelper(self.model_name)


    def generate(self, samples):
        raise NotImplementedError()


    @torch.no_grad()
    def score_candidates(self, samples) -> List[Prediction]:
        prompts: List[str] = []
        prompts_with_mask, candidates, ntrain = self.get_prompts(samples)
        for prompt, cands in zip(prompts_with_mask, candidates):
            prompts.extend(
                [(prompt, candidate) for candidate in cands]
            )
        
        print_r0(f"Average number of train samples: {np.mean(ntrain):.2f}")

        # Predict
        unique_prompts = copy.deepcopy(prompts)
        
        print_r0(f"Predicting {len(samples)} samples with {len(unique_prompts)} prompts..")

        local_hypotheses = self.score_prompts(prompts=unique_prompts)
        
        local_hypotheses = [h[0] for h in local_hypotheses]
        unique_hypotheses = local_hypotheses
        
        utils.assert_equal(len(unique_hypotheses), len(unique_prompts))
        prompt2hypothesis = {
            prompt: hypothesis
            for prompt, hypothesis in zip(unique_prompts, unique_hypotheses)
        }

        hypotheses_with_prompts = [(prompt2hypothesis[prompt], prompt) for prompt in prompts]

        # Score the results
        predictions = self.score_hypotheses(samples, hypotheses_with_prompts)
        
        return predictions


    def score_prompts(self, prompts):
        scored_results = []
        for prompt_id, prompt in enumerate(prompts):
            print_r0(f"prompt_id: {prompt_id}/len(prompts)")
            result = self.helper.call_huggingface_completion(prompt)
            fairseq_result = self.helper.huggingface_result_to_fairseq_result(result)
            print("success!")
            scored_results.append([fairseq_result])
        return scored_results

if __name__ == "__main__":
    print(PREDICTORS_REGISTRY.keys())
