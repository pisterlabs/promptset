from typing import List, Dict, Union, Tuple
import importlib
import datetime
import json
import os
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback, OpenAICallbackHandler
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessage,
    SystemMessagePromptTemplate,
    HumanMessage,
    HumanMessagePromptTemplate,
)
from langchain.prompts import PromptTemplate

from datasets import DatasetDict

from dotenv import load_dotenv

from src.metadata import Metadata
from src.utils import read_yaml

dsbs_du = importlib.import_module("distilling-step-by-step.data_utils")
load_dotenv()


class TeacherQuerier:
    def __init__(self, chat_model: str, dataset_name: str, dataloader: dsbs_du.DatasetLoader, has_valid: bool):
        self.chat_model = ChatOpenAI(model=chat_model, request_timeout=40) if chat_model == "gpt-3.5-turbo" else OpenAI(model=chat_model)
        self.prompt_templates_folder = "./prompt-templates"
        self.dataset_folder = "./datasets"
        self.queries_save_folder = "./query-results"

        self.dataset_name = dataset_name
        self.has_valid = has_valid

        self.metadata = Metadata(dataset_name)
        self.datasets = dataloader.load_from_json(TQ_post_process=True)
    

    def save_querie_results(
        self,
        queries: List[Dict[str, Union[str, int]]],
        split: str,
        prompt_template_id: int,
        save_location: str = None,
        save_name: str = None,
    ) -> None:
        if save_location is None:
            save_location = f"{self.queries_save_folder}/{self.dataset_name}/{split}/{prompt_template_id}"

        if not os.path.exists(save_location):
            os.makedirs(save_location)

        if save_name == "timestamp":
            save_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        else:
            save_name = f"responses.json"

        with open(f"{save_location}/{save_name}", "a") as f:
            for query in queries:
                json.dump(query, f)
                f.write("\n")

    def show_example(self, split: str = "train", idx: int = 0) -> None:
        print(self.datasets[split][idx])

    def build_chain_from_prompt_template(self, prompt_template_id: int) -> LLMChain:
        yaml_file = f"{self.prompt_templates_folder}/{self.dataset_name}.yaml"
        prompt_template = read_yaml(yaml_file)["templates"][prompt_template_id]

        if isinstance(self.chat_model, ChatOpenAI):
            system_message_prompt = SystemMessagePromptTemplate.from_template(prompt_template["system_message"])
            human_message_prompt = HumanMessagePromptTemplate.from_template(prompt_template["user_message"])
            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        elif isinstance(self.chat_model, OpenAI):
            chat_prompt = PromptTemplate.from_template(f'{prompt_template["system_message"]} {prompt_template["user_message"]}')

        chain = LLMChain(llm=self.chat_model, prompt=chat_prompt)

        return chain

    def stringify_chat_prompt(self, formatted_chat_prompt: List[Union[SystemMessage, HumanMessage]]) -> str:
        return f"{formatted_chat_prompt[0].content}\n{formatted_chat_prompt[1].content}"

    def run_chain_with_callback(
        self, chain: LLMChain, template_args: Dict[str, str]
    ) -> Tuple[str, OpenAICallbackHandler]:
        with get_openai_callback() as callback:
            response = chain.run(template_args)	
        return response, callback

    def calculate_batch_query_metrics(self, callbacks: List[OpenAICallbackHandler]) -> Tuple[int, int, float]:
        total_prompt_tokens = sum([callback.prompt_tokens for callback in callbacks])
        total_completion_tokens = sum([callback.completion_tokens for callback in callbacks])
        total_costs = sum([callback.total_cost for callback in callbacks])

        return total_prompt_tokens, total_completion_tokens, total_costs

    def get_already_stored_results(self, split: str, prompt_template_id: int) -> set:
        stored_results = set()
        save_location = f"{self.queries_save_folder}/{self.dataset_name}/{split}/{prompt_template_id}"

        if os.path.exists(f"{save_location}/responses.json"):
            with open(f"{save_location}/responses.json", "r") as f:
                saved_results = [json.loads(line) for line in f]
            for result in saved_results:
                stored_results.add(f"{result['split']}_{result['idx']}_{result['prompt_template_id']}")

        return stored_results

    def batch_query(
        self,
        split: str,
        idxs: List[int],
        prompt_template_id: int,
        template_tuple: List[Tuple[str, str]],
        dont_save: bool = False,
        force_query: bool = False,
        verbosity: int = 2,
    ) -> Tuple[int, int, float]:
        # build prompt template and chain
        chain = self.build_chain_from_prompt_template(prompt_template_id)

        examples = self.datasets[split].select(idxs)
        callbacks = []

        # get already stored results
        stored_results = self.get_already_stored_results(split, prompt_template_id)
        skipped = []

        end = "\r"
        n = 0
        for idx, example in zip(idxs, examples):
            n += 1
            if n == len(idxs): end = "\n"
            if f"{split}_{idx}_{prompt_template_id}" not in stored_results or force_query:
                if verbosity > 0: print(f"QUERYING EXAMPLE {n}/{len(idxs)} ({idx})...", end=end)
                response, callback = self.run_chain_with_callback(
                    chain,
                    {tup[0]: example[tup[1]] for tup in template_tuple}
                )
                callbacks.append(callback)
                # add to stored results in case it was queried in this batch already (should never happen)
                stored_results.add(f"{split}_{idx}_{prompt_template_id}")
                if not dont_save:
                    self.save_querie_results(
                        queries=[
                            {
                                "split": split,
                                "idx": idx,
                                "prompt_template_id": prompt_template_id,
                                "prompt_values": {tup[0]: example[tup[1]] for tup in template_tuple},
                                "complete_prompt": self.stringify_chat_prompt(
                                    chain.prompt.format_messages(**{tup[0]: example[tup[1]] for tup in template_tuple})
                                ) if isinstance(self.chat_model, ChatOpenAI) else chain.prompt.format(**{tup[0]: example[tup[1]] for tup in template_tuple}),
                                "response": response,
                            }
                        ],
                        split=split,
                        prompt_template_id=prompt_template_id,
                    )
            else:
                print(f"SKIPPING EXAMPLE {n}/{len(idxs)} ({idx})...", end=end)
                skipped.append(idx)

        total_prompt_tokens, total_completion_tokens, total_costs = self.calculate_batch_query_metrics(callbacks)
        if verbosity > 1:
            print(
                f"Batch Query completed! (Skipped {len(skipped)} queries as they were already queried and stored.)\nTotal Prompt Tokens: {total_prompt_tokens}\nTotal Completion Tokens: {total_completion_tokens}\nTotal Costs: ${total_costs}"
            )
        # update metadata
        self.metadata.update_from_callback(
            prompt_template_id, total_prompt_tokens, total_completion_tokens, total_costs, len(idxs) - len(skipped)
        )

        return total_prompt_tokens, total_completion_tokens, total_costs

    def query(
        self,
        split: str,
        idx: int,
        prompt_template_id: int,
        template_tuple: List[Tuple[str, str]],
        dont_save: bool = False,
    ) -> None:
        # build prompt template and chain
        chain = self.build_chain_from_prompt_template(prompt_template_id)

        # get example from dataset split
        example = self.datasets[split][idx]
        if isinstance(self.chat_model, ChatOpenAI):
            print(
                self.stringify_chat_prompt(chain.prompt.format_messages(**{tup[0]: example[tup[1]] for tup in template_tuple}))
            )
        elif isinstance(self.chat_model, OpenAI):
            print(chain.prompt.format(**{tup[0]: example[tup[1]] for tup in template_tuple}))
            
        response = chain.run({tup[0]: example[tup[1]] for tup in template_tuple})
        print(f"RESPONSE:\n{response}")

        # save results
        if not dont_save:
            self.save_querie_results(
                queries=[
                    {
                        "split": split,
                        "idx": idx,
                        "prompt_template_id": prompt_template_id,
                        "prompt_values": {tup[0]: example[tup[1]] for tup in template_tuple},
                        "complete_prompt": self.stringify_chat_prompt(
                            chain.prompt.format_messages(**{tup[0]: example[tup[1]] for tup in template_tuple})
                        ),
                        "response": response,
                    }
                ],
                split=split,
                prompt_template_id=prompt_template_id,
            )

    def _batch_query(
        self, split: str, n: int, prompt_template_id: int, dont_save: bool = False, force_query: bool = False
    ) -> Tuple[int, int, float]:
        raise NotImplementedError

    def _query(self, split: str, prompt_template_id: int, dont_save: bool = False) -> None:
        raise NotImplementedError

    def _post_process_data(self, datasets) -> DatasetDict:
        raise NotImplementedError


class ANLITeacherQuerier(TeacherQuerier):
    def __init__(self, chat_model: str = "gpt-3.5-turbo"):
        dataset_name = "anli1"
        has_valid = True
        dataloader = dsbs_du.ANLI1DatasetLoader()
        super().__init__(chat_model, dataset_name, dataloader, has_valid)

    def _batch_query(
        self, split: str, idxs: List[int], prompt_template_id: int, dont_save: bool = False, force_query: bool = False, verbosity: int = 2
    ) -> Tuple[int, int, float]:
        template_tuple = [("premise", "premise"), ("hypothesis", "hypothesis"), ("label", "label")]
        return self.batch_query(split, idxs, prompt_template_id, template_tuple, dont_save, force_query, verbosity)

    def _query(self, split: str, idx: int, prompt_template_id: int, dont_save: bool = False) -> None:
        template_tuple = [("premise", "premise"), ("hypothesis", "hypothesis"), ("label", "label")]
        self.query(split, idx, prompt_template_id, template_tuple, dont_save)

    def _post_process_data(self, datasets) -> DatasetDict:
        def label_idx2text(example):
            if example["label"] == 0:
                example["label"] = "entailment"
            elif example["label"] == 1:
                example["label"] = "neutral"
            elif example["label"] == 2:
                example["label"] = "contradiction"
            return example

        datasets = datasets.map(label_idx2text)
        datasets = datasets.remove_columns(["reason"])

        return datasets


class CQATeacherQuerier(TeacherQuerier):
    def __init__(self, chat_model: str = "gpt-3.5-turbo"):
        dataset_name = "cqa"
        has_valid = False
        dataloader = dsbs_du.CQADatasetLoader()
        super().__init__(chat_model, dataset_name, dataloader, has_valid)

    def _batch_query(
        self, split: str, idxs: List[int], prompt_template_id: int, dont_save: bool = False, force_query: bool = False, verbosity: int = 2
    ) -> Tuple[int, int, float]:
        template_tuple = [
            ("question", "question"),
            ("choice_a", "c_0"),
            ("choice_b", "c_1"),
            ("choice_c", "c_2"),
            ("choice_d", "c_3"),
            ("choice_e", "c_4"),
        ]
        return self.batch_query(split, idxs, prompt_template_id, template_tuple, dont_save, force_query, verbosity)

    def _query(self, split: str, idx: int, prompt_template_id: int, dont_save: bool = False) -> None:
        template_tuple = [
            ("question", "question"),
            ("choice_a", "c_0"),
            ("choice_b", "c_1"),
            ("choice_c", "c_2"),
            ("choice_d", "c_3"),
            ("choice_e", "c_4"),
        ]
        self.query(split, idx, prompt_template_id, template_tuple, dont_save)

    def _post_process_data(self, datasets):
        def prepare_input(example):
            example["c_0"] = example["choices"][0]
            example["c_1"] = example["choices"][1]
            example["c_2"] = example["choices"][2]
            example["c_3"] = example["choices"][3]
            example["c_4"] = example["choices"][4]

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(["abstractive_explanation", "extractive_explanation"])

        return datasets


class ESNLITeacherQuerier(TeacherQuerier):
    def __init__(self, chat_model: str = "gpt-3.5-turbo"):
        dataset_name = "esnli"
        has_valid = True
        dataloader = dsbs_du.ESNLIDatasetLoader(subset="small")
        super().__init__(chat_model, dataset_name, dataloader, has_valid)

    def _batch_query(
        self, split: str, idxs: List[int], prompt_template_id: int, dont_save: bool = False, force_query: bool = False, verbosity: int = 2
    ) -> Tuple[int, int, float]:
        template_tuple = [("premise", "premise"), ("hypothesis", "hypothesis")]
        return self.batch_query(split, idxs, prompt_template_id, template_tuple, dont_save, force_query, verbosity)

    def _query(self, split: str, idx: int, prompt_template_id: int, dont_save: bool = False) -> None:
        template_tuple = [("premise", "premise"), ("hypothesis", "hypothesis")]
        self.query(split, idx, prompt_template_id, template_tuple, dont_save)

    def _post_process_data(self, datasets):
        def label_idx2text(example):
            if example["label"] == 0:
                example["label"] = "entailment"
            elif example["label"] == 1:
                example["label"] = "neutral"
            elif example["label"] == 2:
                example["label"] = "contradiction"

            return example

        datasets = datasets.map(label_idx2text)
        datasets = datasets.remove_columns(["explanation_1", "explanation_2", "explanation_3"])

        return datasets


class SVAMPTeacherQuerier(TeacherQuerier):
    def __init__(self, chat_model: str = "gpt-3.5-turbo"):
        dataset_name = "svamp"
        has_valid = False
        dataloader = dsbs_du.SVAMPDatasetLoader()
        super().__init__(chat_model, dataset_name, dataloader, has_valid)

    def _batch_query(
        self, split: str, idxs: List[int], prompt_template_id: int, dont_save: bool = False, force_query: bool = False, verbosity: int = 2
    ) -> Tuple[int, int, float]:
        template_tuple = [("question", "input")]
        return self.batch_query(split, idxs, prompt_template_id, template_tuple, dont_save, force_query, verbosity)

    def _query(self, split: str, idx: int, prompt_template_id: int, dont_save: bool = False) -> None:
        template_tuple = [("question", "input")]
        self.query(split, idx, prompt_template_id, template_tuple, dont_save)

    def _post_process_data(self, datasets):
        return datasets
