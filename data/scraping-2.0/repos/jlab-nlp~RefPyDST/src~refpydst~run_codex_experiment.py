import argparse
import copy
import json
import logging
import os
import sys
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import wandb
from openai.error import RateLimitError, APIError, OpenAIError
from refpydst.data_types import ExampleListDecoderConfig, CodexDecodingConfig
from refpydst.data_types import Turn, CompletionParser, CodexPromptingRunConfig

from refpydst.artifacts import output_dir_to_run_or_artifact_name
from refpydst.codex_client import CodexClient, PromptOverlengthError
from refpydst.db.ontology import Ontology
from refpydst.generation_experiment import AbstractLMPromptingExperiment
from refpydst.prompting import PROMPT_VARIANTS, PromptGenerator, STOP_SEQUENCES, IC_DST
from refpydst.retriever.abstract_example_retriever import ExampleRetriever
from refpydst.utils.general import read_json, subtract_dict, get_output_dir_full_path, WANDB_ENTITY, WANDB_PROJECT
from refpydst.utils.state_recorder import PreviousStateRecorder


class CodexExperiment(AbstractLMPromptingExperiment):
    """
    Class for managing an experiment with Codex
    """

    test_set: List[Turn]
    use_gold: bool
    prompt_format: str
    only_use_turn_idx: int
    ontology: Ontology
    prediction_recorder: PreviousStateRecorder
    prompt_generator: PromptGenerator
    retriever: ExampleRetriever
    num_examples: int
    demonstration_mapping: Optional[Dict[str, Dict[str, List[Turn]]]]
    train_set: List[Turn]
    completion_parser: CompletionParser
    codex_client: CodexClient
    mwz_ver: str
    num_distinct_demonstrations: Optional[int]
    lm_decoding_config: CodexDecodingConfig
    min_null_sequence_log_probability: float
    min_null_token_log_probability: float

    def __init__(self, test_set_path: str, ontology_file_path: str, use_gold: bool = False,
                 prompt_format: str = None, turn: int = -1, num_examples: int = 10, retriever_dir: str = None,
                 train_set_path: str = None, demonstration_mapping_path: str = None,
                 codex_engine: str = "code-davinci-002", mwz_ver: str = "2.4",
                 retriever_type: str = None, decoder_config: Optional[ExampleListDecoderConfig] = None,
                 lm_decoding_config: Optional[CodexDecodingConfig] = None,
                 format_example: Optional[Turn] = None,
                 **kwargs) -> None:
        super().__init__(
            test_set_path=test_set_path,
            ontology_file_path=ontology_file_path,
            use_gold=use_gold,
            prompt_format=prompt_format,
            turn=turn,
            num_examples=num_examples,
            format_example=format_example,
            retriever_dir=retriever_dir,
            train_set_path=train_set_path,
            demonstration_mapping_path=demonstration_mapping_path,
            mwz_ver=mwz_ver,
            retriever_type=retriever_type,
            decoder_config=decoder_config,
            **kwargs
        )
        self.lm_decoding_config = lm_decoding_config
        
        min_null_token_prob: float = self.lm_decoding_config and self.lm_decoding_config.get('min_token_null_probability', 0) or 0
        self.min_null_token_log_probability = np.log(min_null_token_prob) if min_null_token_prob != 0 else sys.float_info.min
        min_null_sequence_prob: float = self.lm_decoding_config and self.lm_decoding_config.get('min_null_probability', 0) or 0
        self.min_null_sequence_log_probability = np.log(min_null_sequence_prob) if min_null_sequence_prob != 0 else sys.float_info.min
        self.codex_client = CodexClient(engine=codex_engine, stop_sequences=STOP_SEQUENCES.get(self.prompt_format))

    def generate_completion(self, prompt_text: str, data_item: Turn, examples: List[Turn]) -> Tuple[
        Dict[str, float], List[Turn]]:
        # codex completion
        complete_flag = False
        parse_error_count = 0
        completions: Dict[str, float] = {}
        while not complete_flag and parse_error_count < 5:
            try:
                if self.lm_decoding_config is None or self.lm_decoding_config.get("method", "greedy") == "greedy":
                    completions = self.codex_client.greedy_lm_completion(prompt_text)
                elif self.lm_decoding_config["method"] == "top_p":
                    completions = self.codex_client.top_p_lm_completion(prompt_text, **self.lm_decoding_config)
                elif self.lm_decoding_config["method"] == "mutual_info":
                    # below gives completions and log probabilities for top-p based inputs
                    completions = self.codex_client.top_p_lm_completion(prompt_text, **self.lm_decoding_config)
                    self.logger.log({"completions": len(completions), "rescoring": 1 if len(completions) > 1 else 0})
                    if len(completions) > 1:
                        # re-score these according to mutual information by dividing by our 'null' prompt:
                        max_to_consider: int = self.lm_decoding_config['max_mi_candidates']
                        null_prompt_text: str = self.prompt_generator.get_prompt(
                            data_item,
                            examples=examples,
                            n_examples=len(examples),
                            prompt_format=self.lm_decoding_config['null_prompt_format']
                        )
                        token_log_prob_given_null: Dict[str, List[float]] = {}
                        completion_to_canonical: Dict[str, str] = {}
                        # only consider the top-K possible completions, sorted by log-probability
                        candidates = sorted(completions, key=completions.get, reverse=True)[:max_to_consider]
                        i: int = 0
                        while len(token_log_prob_given_null) < len(candidates):
                            try:
                                completion = candidates[i]
                                # grammatical/ontology-supported completions have highest probability given prompt,
                                # but low-probability ones still occur in top-p sampling. Under a null prompt, these can
                                # have arbitrarily low probability and thus high mutual information. However, we want to
                                # consider these illegal completions as having zero probability under both prompts, so
                                # we skip them here (this technically does not zero-out their probability under main
                                # prompt, but not a significant issue in practice w/ Codex.)
                                predicted_context = self.prediction_recorder.retrieve_previous_turn_state(data_item)
                                checkable_parse = None
                                try:
                                    checkable_parse = self.normalizer.normalize(
                                        self.completion_parser(completion, predicted_context,
                                                               exceptions_are_empty=False)
                                    )
                                except Exception as e:
                                    print("got exception while parsing. Continuing without a checkable_parse", e)

                                if checkable_parse is not None and all(self.ontology.is_in_ontology(slot, value)
                                                                       for slot, value in checkable_parse.items()):
                                    # the completion as returned from Codex is not necessarily the one we want to use to
                                    # check the null log probability. For instance if it includes a value that we
                                    # normalized out in the ontology, it likely repeats something unexpected from the
                                    # user utterance, like a mis-spelling, and will have extremely low null-prompt
                                    # probability, e.g:
                                    # 'can you book me a stay at the acron guest houe'
                                    # =>
                                    # hotel = agent.find_hotel(name="acron guest houe") <== in-context examples likely
                                    # not mis-spelled, so in null prompt these have very low probability
                                    # if we don't use 'canonical__completions', send it back as is. Else, canonicalize
                                    # from parse we derived:
                                    null_prompt_completion: str = completion
                                    if self.lm_decoding_config.get('canonical_completions'):
                                        null_prompt_completion: str = self.prompt_generator.get_canonical_completion(
                                            checkable_parse, predicted_context, turn=data_item,
                                            prompt_format=self.prompt_format
                                        )
                                    completion_to_canonical[completion] = null_prompt_completion
                                    logging.info(
                                        f"Using {null_prompt_completion} as the canonical completion for {completion}")
                                    null_prompt_completion_log_probs: List[float] = self.codex_client.get_completion_log_probabilities(
                                        null_prompt_text,
                                        null_prompt_completion.strip() + self.lm_decoding_config.get("stop_token", ";"),
                                    )
                                    token_log_prob_given_null[completion] = null_prompt_completion_log_probs
                                else:
                                    token_log_prob_given_null[completion] = [0]  # p(null_prompt_completion) == 1
                                i += 1
                            except (RateLimitError, APIError) as e:
                                print(e)  # retry

                        # Our estimate of completion probability can be arbitrarily low for a poorly calibrated null
                        # prompt. Thus we'll instead have a prior hyper-parameter which indicates the minimum null log
                        # probability we'll consider. However, we'll log the original null log probability, so that we
                        # can tune this hyper-parameter based on the distribution of these values.
                        raw_log_prob_given_null: Dict[str, float] = {k: sum(v) for k, v in token_log_prob_given_null.items()}
                        data_item['log_prob_given_null'] = copy.deepcopy(raw_log_prob_given_null)
                        data_item['log_probs'] = copy.deepcopy(completions)
                        data_item['completion_to_canonical'] = completion_to_canonical
                        data_item['token_log_prob_given_null'] = copy.deepcopy(token_log_prob_given_null)

                        # re-weight according mutual information. Treat missing values as having a denominator of 1, so
                        # minus log(1) = 0. Enforce our minimum acceptable probability given the null prompt
                        self.logger.log({
                            "min_log_prob_given_null": min(v for v in raw_log_prob_given_null.values()),
                            "max_log_prob_given_null": max(v for v in raw_log_prob_given_null.values()),
                            "mean_log_prob_given_null": np.mean([v for v in raw_log_prob_given_null.values()]),
                        })
                        # first: clip token log-probabilities if that parameter is set
                        processed_token_log_probs: Dict[str, List[float]] = {
                            k: [max(log_p_token, self.min_null_token_log_probability) for log_p_token in v]
                            for k, v in token_log_prob_given_null.items()
                        }
                        # next: clip sequence probs if that parameter is set (should just be using one of these)
                        final_log_prob_given_null: Dict[str, float] = {k: max(sum(v), self.min_null_sequence_log_probability) for k, v in
                                                                       processed_token_log_probs.items()}
                        if self.lm_decoding_config.get('null_prompt_weight', 1.) != 1:
                            beta: float = self.lm_decoding_config['null_prompt_weight']
                            final_log_prob_given_null = {k: beta * v for k, v in final_log_prob_given_null.items()}
                        completions = subtract_dict(completions, final_log_prob_given_null, default_value=0)
                    else:
                        self.logger.log({
                            "min_log_prob_given_null": 0,
                            "max_log_prob_given_null": 0,
                            "mean_log_prob_given_null": 0,
                        })
                else:
                    raise ValueError(f"Unsupported decoding arguments: {self.lm_decoding_config}")
            except PromptOverlengthError as e:
                logging.warning(e)
                logging.info("prompt overlength, retrying with fewer examples")
                examples = examples[1:]
                prompt_text = self.get_prompt_text(data_item=data_item, examples=examples)
            except ValueError as e:
                logging.exception(e)
                raise e
            except (RateLimitError, APIError, OpenAIError) as e:
                # client will manage sleeping/timing
                logging.exception(e)
            except BaseException as e:
                logging.exception(e)
                if type(e) == KeyboardInterrupt:
                    raise e
            else:
                # interesting python idiom: try/except/else: else executes if there is no exception of any typ e in the
                # try block (kind of un-needed here, but can be used with finally to be try/except/else/finally)
                try:
                    # check if CODEX is crazy
                    predicted_context = self.prediction_recorder.retrieve_previous_turn_state(data_item)
                    # for now, just verify our best completion is parse-able
                    temp_parse = self.completion_parser(max(completions, key=completions.get), predicted_context)
                    complete_flag = True
                except Exception:
                    parse_error_count += 1

        if not complete_flag:
            raise ValueError("unable to generate completion")
        return completions, examples


def main(train_fn: str, retriever_dir: str, output_dir: str, test_fn: str, prompt_format: str = IC_DST,
         mwz_ver: str = "2.4",
         codex_engine: str = "code-davinci-002", demonstration_mapping_path: str = None,
         retriever_type: str = "EmbeddingRetriever", decoder_config: ExampleListDecoderConfig = None,
         lm_decoding_config: Optional[CodexDecodingConfig] = None,
         artifact_cache: str = None,
         format_example: Optional[Turn] = None, num_examples: int = 10, **kwargs) -> None:
    # create the output folder
    os.makedirs(output_dir, exist_ok=True)
    # write out this experiment's configuration
    exp_config: Dict[str, Union[str, int]] = dict(locals())
    with open(os.path.join(output_dir, "exp_config.json"), 'w') as f:
        json.dump(exp_config, f, indent=4)

    # read the ontology and the test set
    ontology_file_path = f"db/multiwoz/{mwz_ver}/ontology.json"
    if mwz_ver == '2.1':
        test_set_path = test_fn or "./data/mw21_100p_test.json"
    else:
        test_set_path = test_fn or "./data/mw24_100p_test.json"

    experiment: CodexExperiment = CodexExperiment(
        artifact_cache=artifact_cache,
        train_set_path=train_fn,
        retriever_dir=retriever_dir,
        test_set_path=test_set_path,
        ontology_file_path=ontology_file_path,
        num_examples=num_examples,
        prompt_format=prompt_format,
        demonstration_mapping_path=demonstration_mapping_path,
        codex_engine=codex_engine,
        mwz_ver=mwz_ver,
        retriever_type=retriever_type,
        decoder_config=decoder_config,
        lm_decoding_config=lm_decoding_config,
        output_dir=output_dir,
        format_example=format_example,
        **kwargs
    )

    try:
        running_log, stats = experiment.run()
    finally:
        artifact: wandb.Artifact = wandb.Artifact(wandb.run.name, type="run_output")
        artifact.add_dir(experiment.output_dir)
        wandb.log_artifact(artifact)

    # write out running log
    with open(os.path.join(output_dir, "running_log.json"), 'w') as f:
        json.dump(running_log, f)

    if len(running_log) == len(experiment.test_set):
        run = wandb.Api().run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
        run.tags.append("complete_run")
        run.update()


if __name__ == "__main__":
    if os.path.exists(sys.argv[1]):
        run_file: str = sys.argv[1]
        # arguments are input from a configuration file if the first argument to the program is a valid file
        args: CodexPromptingRunConfig = read_json(run_file)
        if 'output_dir' not in args:
            args['output_dir'] = get_output_dir_full_path(run_file.replace('.json', ''))
        if not 'run_name' in args:
            args['run_name'] = output_dir_to_run_or_artifact_name(args['output_dir'])
    else:
        # otherwise, try to parse from argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_fn', type=str, help="training data file (few-shot or full shot)",
                            required=True)  # e.g. "./data/mw21_10p_train_v3.json"
        parser.add_argument('--prompt_format', type=str, choices=PROMPT_VARIANTS,
                            help=f"prompt format variant, among: {', '.join(PROMPT_VARIANTS)}",
                            default="IC-DST")  # e.g. "IC-DST"
        parser.add_argument('--retriever_dir', type=str, required=True,
                            help="sentence transformer saved path")  # "./retriever/expts/mw21_10p_v3_0304_400_20"
        parser.add_argument('--output_dir', type=str, default="./expts/debug",
                            help="directory to save running log and configs")
        parser.add_argument('--mwz_ver', type=str, default="2.1", choices=['2.1', '2.4'], help="version of MultiWOZ")
        parser.add_argument('--codex_engine', type=str, default="code-davinci-002", choices=["text-davinci-002"],
                            help="version of GPT-3/Codex to complete with")
        parser.add_argument('--demonstration_mapping_path', type=str, default=None,
                            help="if provided, don't use retriever to find nearby dialogue turns, and instead use those "
                                 "provided in the mapping load-able at this path. It should contain a dictionary of the"
                                 "form: {dial_id: {turn_id: [(dial_id, turn_id), ...]}, ...}")
        parser.add_argument('--test_fn', type=str, default='', help="file to evaluate on, empty means use the test set")
        parser.add_argument('--retriever_type', type=str, default='EmbeddingRetriever',
                            help="what kind of retriever to use")
        args = parser.parse_args()
        args = vars(args)
    default_run_name: str = output_dir_to_run_or_artifact_name(args['output_dir'])
    default_run_group: str = default_run_name.rsplit('-', maxsplit=1)[0]
    wandb_entity: str = os.environ.get(WANDB_ENTITY, "kingb12")
    wandb_project: str = os.environ.get(WANDB_PROJECT, "refpydst")
    run = wandb.init(config=args, project=wandb_project, entity=wandb_entity,
                     name=args.get("run_name", default_run_name), notes=args.get("run_notes", None),
                     group=args.get("run_group", default_run_group),
                     tags=args.get("run_tags", None))
    main(**args)
