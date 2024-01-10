"""
generate domain-specific instruction-tuning data
"""
import copy
import shutil
import time
import json
import os
import random
import re
import string
import logging
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple, Dict, Union, Optional, Any
import argparse

import numpy as np
import tqdm
from rouge_score import rouge_scorer
# import utils
import openai
import fire
from transformers import GPT2TokenizerFast, AutoTokenizer, BertTokenizer
from anytree import AnyNode, Node, PreOrderIter, LevelOrderIter
from anytree.importer import DictImporter, JsonImporter
from anytree.exporter import JsonExporter, DictExporter
import asyncio

level = logging.DEBUG
# level = logger.INFO
logger = logging.getLogger(__name__)
logger.setLevel(level)
c_handler = logging.StreamHandler()
c_handler.setLevel(level)
c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)

# Add handlers to the logger
logger.addHandler(c_handler)

meta_info_filename = "tree_meta_info.json"

openai.api_key = "YOUR OPENAI API KEY"
claude_key = "YOUR CLAUDE API KEY"


class DomainTree(object):
    """docstring for DomainTree"""

    def __init__(
            self,
            root: AnyNode,
            unique_notd_id: str = "task_name",
            name_to_node: Dict[str, AnyNode] = None,
            **kwargs
    ):
        self.root = root
        self.unique_id = unique_notd_id
        self.name_to_node = name_to_node if name_to_node is not None \
            else {getattr(node, self.unique_id): node for node in PreOrderIter(self.root)}

        # TODO: load from config file instead of hard-coded
        self.prepare_mine_hparams(**kwargs)
        self.prepare_prompt()
        self.prepare_tools()

    def extend_node_children(
            self,
            node_to_extend: Union[AnyNode, str],
            extend_num: int = None,
            extend_batch_size: int = None,
    ) -> List[AnyNode]:
        all_new_nodes = list()
        while len(all_new_nodes) < extend_num:
            gap = extend_num - len(all_new_nodes)
            logger.info(f"Already extended num: {len(all_new_nodes)}. This time gap: {gap}")
            new_nodes: List[AnyNode] = self._extend_node_children(
                node_to_extend,
                extend_num=max(gap, self.min_extend_num),
                extend_batch_size=extend_batch_size,
            )
            all_new_nodes += new_nodes
        return all_new_nodes

    def _extend_node_children(
            self,
            node_to_extend: Union[AnyNode, str],
            extend_num: int = None,
            extend_batch_size: int = None,
    ) -> List[AnyNode]:
        """
        Extending the given node's children
        """
        extend_num = self.default_extend_num if extend_num is None else extend_num
        extend_batch_size = self.default_extend_batch_size if extend_batch_size is None else extend_batch_size
        # locate the node
        if type(node_to_extend) is str:
            node_to_extend: AnyNode = self.name_to_node[node_to_extend]
        node_name = getattr(node_to_extend, self.unique_id)
        current_scenario = f"extend_node_children for {node_name}"
        logger.info(current_scenario)
        # check its existing children
        existing_children = list(node_to_extend.children)
        existing_children_names = [getattr(child, self.unique_id) for child in node_to_extend.children]
        if len(existing_children) >= self.max_child_num:
            logger.warning(f"Failed trial to extend node {node_name}: already having {len(existing_children)} children")
            return []
        if len(existing_children) + extend_num > self.max_child_num:
            logger.warning(f"Exceeding max_child_num if extending by {extend_num}, using remaining num instead.")
            extend_num = self.max_child_num - len(existing_children)
        # formulate the prompt and api request
        base_prompt = self.extend_node_prompt
        existing_siblings = list(node_to_extend.siblings)
        if extend_batch_size > 1:
            prompt = []
            for _ in range(extend_batch_size):
                demonstrate_examples = self.get_demonstrate_examples(node_to_extend)
                prompt_tmp = self.encode_prompt(
                    base_prompt=base_prompt,
                    demonstrate_examples=demonstrate_examples,  # Example triplet
                    target_task=node_name,  # name of node_to_extend
                    existing_children=existing_children,  # list of children of node_to_extend
                    existing_siblings=existing_siblings,  # list of siblings of node_to_extend
                    num_examples_per_time=self.num_example_extend,  # num of triplet for each new mined subtask
                    extend_num=extend_num,
                    # new_subtask=new_subtask,  # only used when enriching nodes
                    # new_subtask_reason=new_subtask_reason,  # only used when enriching nodes
                    target_children_num=self.max_child_num,
                )
                prompt.append(prompt_tmp)
        else:
            demonstrate_examples = self.get_demonstrate_examples(node_to_extend)
            prompt = self.encode_prompt(
                base_prompt=base_prompt,
                demonstrate_examples=demonstrate_examples,  # Example triplet
                target_task=node_name,  # name of node_to_extend
                existing_children=existing_children,  # list of children of node_to_extend
                existing_siblings=existing_siblings,  # list of siblings of node_to_extend
                num_examples_per_time=self.num_example_extend,  # num of triplet for each new mined subtask
                extend_num=extend_num,
                # new_subtask=new_subtask,  # only used when enriching nodes
                # new_subtask_reason=new_subtask_reason,  # only used when enriching nodes
                target_children_num=self.max_child_num,
            )
        logger.debug(f"Num of existing_children: {len(existing_children)}")
        logger.debug(f"Num of existing_siblings: {len(existing_siblings)}")
        logger.debug(f"Final prompt: {prompt}")

        logger.info(f"{self.assistant_name}: Online querying...")
        result = self.request_func(prompt)
        logger.info("Received online querying results")
        logger.debug(f"raw request result: {result}")
        if result is None:
            logger.warning(f"Received error response!!")
            return []
        if extend_batch_size > 1:
            for p, r in zip(prompt, result):
                self.write_query_log(p, r)
        else:
            self.write_query_log(prompt, result)

        # new_subtask, new_subtask_reason, new_instructions = \
        #     self.post_process_gpt3_response_extend(result)
        if extend_batch_size > 1:
            new_subtask, new_subtask_reason, new_instructions = [], [], []
            for r in result:
                new_subtask_tmp, new_subtask_reason_tmp, new_instructions_tmp = self.postprocess_extend(r)
                new_subtask += new_subtask_tmp
                new_subtask_reason += new_subtask_reason_tmp
                new_instructions += new_instructions_tmp
        else:
            new_subtask, new_subtask_reason, new_instructions = \
                self.postprocess_extend(result)
        logger.debug(f"len(new_subtask): {len(new_subtask)}; new_subtask: {new_subtask};")

        add_nodes = list()
        if not (len(new_subtask) == len(new_subtask_reason) == len(new_instructions)):
            logger.warning(f"{current_scenario}, encountering bad completion result")
            if extend_batch_size > 1:
                for p, r in zip(prompt, result):
                    self.handle_bad_completion(r,
                                               prompt=p,
                                               request_scenario=current_scenario, )
            else:
                self.handle_bad_completion(
                    result,
                    prompt=prompt,
                    request_scenario=current_scenario,
                )
            return add_nodes

        # save file and update tree
        for subtask, reason, examples in zip(new_subtask, new_subtask_reason, new_instructions):
            subtask_id = self.formalize_taskname(subtask)
            # node_save_file = os.path.join(self.general_outdir, f"{subtask_id}.json")
            node_info = {
                self.unique_id: subtask_id,
                "raw_task_name": subtask,
                "parent": node_to_extend,
                "reason": reason,
                "examples": examples,
                # "config_file": node_save_file,
                "config_filename": f"{subtask_id}.json",

            }
            new_node = self.add_node(node_info)
            add_nodes.append(new_node)
        return add_nodes

    def write_query_log(self, prompt: str, res: Dict[str, str]):
        with open(os.path.join(self.general_outdir, "query_log.jsonl"), "a+", encoding="utf-8") as f_out:
            query_log = {
                "prompt": prompt,
                "res": res,
            }
            json.dump(
                query_log,
                f_out,
                ensure_ascii=False,
            )
            f_out.write("\n")

    def enrich_node_samples(
            self,
            node_to_enrich: AnyNode,
            enrich_num: int = None,
            enrich_batch_size: int = None,
    ) -> List[Dict[str, str]]:
        all_new_examples = list()
        while len(all_new_examples) < enrich_num:
            gap = enrich_num - len(all_new_examples)
            logger.info(f"Already enriched example num: {len(all_new_examples)}. This time gap: {gap}")
            new_examples: List[Dict[str, str]] = self._enrich_node_samples(
                node_to_enrich,
                enrich_num=max(gap, self.min_extend_num),
                enrich_batch_size=enrich_batch_size
            )
            all_new_examples += new_examples
        return all_new_examples

    def _enrich_node_samples(
            self,
            node_to_enrich: AnyNode,
            enrich_num: int = None,
            enrich_batch_size: int = None,
    ) -> List[Dict[str, str]]:
        """
        Given a mined node, add more <instruct, input, output> belonging to this node
        """
        enrich_num = self.default_enrich_num if enrich_num is None else enrich_num
        enrich_batch_size = self.default_enrich_batch_size if enrich_batch_size is None else enrich_batch_size
        # get existing num of example
        if type(node_to_enrich) is str:
            node_to_enrich: AnyNode = self.name_to_node[node_to_enrich]
        new_subtask = node_name = getattr(node_to_enrich, self.unique_id)

        current_scenario = f"enrich_node_samples for {node_name}"
        logger.info(current_scenario)

        new_subtask_reason = getattr(node_to_enrich, "reason", "")
        # hard-coded to handle enriching root node
        parent_node = node_to_enrich.parent if node_to_enrich.parent is not None else node_to_enrich

        parent_node_name = getattr(parent_node, self.unique_id)
        existing_examples = getattr(node_to_enrich, "examples", [])
        if len(existing_examples) >= self.max_example_num:
            logger.warning(f"Failed trial to enrich node {node_name}: already having {len(existing_examples)} examples")
            return
        if len(existing_examples) + enrich_num > self.max_example_num:
            logger.warning(f"Exceeding max_child_num if extending by {enrich_num}, using remaining num instead.")
            enrich_num = self.max_example_num - len(existing_examples)
        base_prompt = self.enrich_node_prompt
        new_instructions = list()
        existing_children = list(parent_node.children)
        existing_siblings = list(parent_node.siblings)
        if enrich_batch_size > 1:
            prompt = []
            for _ in range(enrich_batch_size):
                demonstrate_examples = self.get_demonstrate_examples(node_to_enrich)
                # only sampling from the seed tasks
                prompt_tmp = self.encode_prompt(
                    base_prompt=base_prompt,
                    demonstrate_examples=demonstrate_examples,  # Example triplet
                    target_task=parent_node_name,  # name of node_to_extend
                    existing_children=existing_children,  # list of children of node_to_extend,
                    existing_siblings=existing_siblings,  # list of siblings of node_to_extend
                    num_examples_per_time=self.num_example_enrich,  # num of triplet for each new mined subtask
                    new_subtask=new_subtask,  # only used when enriching nodes
                    new_subtask_reason=new_subtask_reason,  # only used when enriching nodes
                    # extend_num=extend_num,  # only used when extending nodes
                    # target_children_num=self.max_child_num,  # only used when extending nodes
                )
                prompt.append(prompt_tmp)
        else:
            demonstrate_examples = self.get_demonstrate_examples(node_to_enrich)
            # only sampling from the seed tasks
            prompt = self.encode_prompt(
                base_prompt=base_prompt,
                demonstrate_examples=demonstrate_examples,  # Example triplet
                target_task=parent_node_name,  # name of node_to_extend
                existing_children=existing_children,  # list of children of node_to_extend,
                existing_siblings=existing_siblings,  # list of siblings of node_to_extend
                num_examples_per_time=self.num_example_enrich,  # num of triplet for each new mined subtask
                new_subtask=new_subtask,  # only used when enriching nodes
                new_subtask_reason=new_subtask_reason,  # only used when enriching nodes
                # extend_num=extend_num,  # only used when extending nodes
                # target_children_num=self.max_child_num,  # only used when extending nodes
            )
        logger.debug(f"Final prompt: {prompt}")
        logger.info(f"{self.assistant_name}: Online querying...")
        # completion = self.request_openai(prompt)
        # result = completion['choices'][0]
        result = self.request_func(prompt)
        logger.info("Received online querying results")
        if enrich_batch_size > 1:
            for p, r in zip(prompt, result):
                self.write_query_log(p, r)
        else:
            self.write_query_log(prompt, result)
        logger.debug(f"raw request result: {result}")
        # _, _, new_instructions = \
        #     self.post_process_gpt3_response_enrich(result, new_subtask, new_subtask_reason)
        if enrich_batch_size > 1:
            new_instructions = []
            for r in result:
                _, _, new_instructions_tmp = self.postprocess_enrich(r, new_subtask, new_subtask_reason)
                new_instructions += new_instructions_tmp
        else:
            _, _, new_instructions = \
                self.postprocess_enrich(result, new_subtask, new_subtask_reason)

        # update node config
        node_to_enrich.examples = existing_examples + new_instructions
        self.update_file(node_to_enrich, )
        return new_instructions

    def request_openai_single(self, prompt: str) -> Dict[str, str]:
        response = None
        for trial_idx in range(self.max_retry_times):
            try:
                messages = [
                    {"role": "user",
                     "content": prompt},
                ]
                prompt_len = len(self.tokenizer(prompt)['input_ids'])
                completion = openai.ChatCompletion.create(
                    messages=messages,
                    max_tokens=4096 - 300 - prompt_len,
                    **self.openai_kwargs,
                )
                result = completion['choices'][0]
                response = {
                    "raw_response": result.get("message", {}).get("content", ""),
                    "stop_reason": result.get("finish_reason", ),
                }
                return response
            except Exception as e:
                logger.warning(str(e))
                logger.warning(f"Trail No. {trial_idx + 1} Failed, now sleep and retrying...")
                time.sleep(self.request_sleep_time)
        return response

    async def dispatch_openai_requests(
            self,
            messages_list: List[List[Dict[str, Any]]],
            model: str,
            temperature: float,
            max_tokens: int,
            top_p: float,
            n: int,
            logit_bias: dict,
    ) -> List[str]:
        """Dispatches requests to OpenAI API asynchronously.

        Args:
            messages_list: List of messages to be sent to OpenAI ChatCompletion API.
            model: OpenAI model to use.
            temperature: Temperature to use for the model.
            max_tokens: Maximum number of tokens to generate.
            top_p: Top p to use for the model.
            n: Return sentence nums.
            logit_bias: logit bias.
        Returns:
            List of responses from OpenAI API.
        """
        async_responses = [
            openai.ChatCompletion.acreate(
                model=model,
                messages=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
                logit_bias=logit_bias,
            )
            for x in messages_list
        ]
        return await asyncio.gather(*async_responses)

    def request_openai_dispatch(self, prompt_list: List[str]) -> List[Dict[str, str]]:
        responses = None
        for trial_idx in range(self.max_retry_times):
            try:
                messages_list = [[
                    {"role": "user",
                     "content": prompt},
                ] for prompt in prompt_list]
                prompt_len = max([len(self.tokenizer(prompt)['input_ids']) for prompt in prompt_list])
                completions = asyncio.run(
                    self.dispatch_openai_requests(
                        messages_list=messages_list,
                        max_tokens=4096 - 300 - prompt_len,
                        **self.openai_kwargs,
                    )
                )
                responses = []
                for completion in completions:
                    result = completion['choices'][0]
                    response = {
                        "raw_response": result.get("message", {}).get("content", ""),
                        "stop_reason": result.get("finish_reason", ),
                    }
                    responses.append(response)
                return responses
            except Exception as e:
                logger.warning(str(e))
                logger.warning(f"Trail No. {trial_idx + 1} Failed, now sleep and retrying...")
                time.sleep(self.request_sleep_time)
        return responses

    def request_openai(self, prompt):
        use_async = not type(prompt) is str
        if use_async:
            response = self.request_openai_dispatch(prompt)
        else:
            response = self.request_openai_single(prompt)
        return response

    def request_claude(self, prompt: str) -> Dict[str, str]:
        import anthropic
        result = None
        for trial_idx in range(self.max_retry_times):
            try:
                prompt = f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"
                prompt_len = len(self.tokenizer(prompt)['input_ids'])
                resp = self.claude_client.completion(
                    prompt=prompt,
                    max_tokens_to_sample=4096 - 300 - prompt_len,
                    **self.claude_kwargs,
                )
                result = {
                    "raw_response": resp["completion"],
                    "stop_reason": resp["stop_reason"],
                }
                return result
            except Exception as e:
                logger.warning(str(e))
                logger.warning(f"Trail No. {trial_idx + 1} Failed, now sleep and retrying...")
                time.sleep(self.request_sleep_time)
        return result

    def prepare_mine_hparams(self, **kwargs):
        """
        Currently all hard-coded
        """

        self.assistant_name = kwargs.get("assistant_name") if kwargs.get("assistant_name") else "openai"
        self.assistant_request_map = {
            "openai": self.request_openai,
            "claude": self.request_claude,
        }
        self.assistant_postprocess_map = {
            "openai": (self.post_process_gpt3_response_extend, self.post_process_gpt3_response_enrich),
            "claude": (self.post_process_gpt3_response_extend, self.post_process_gpt3_response_enrich),
        }
        self.postprocess_extend, self.postprocess_enrich = self.assistant_postprocess_map[self.assistant_name]
        self.request_func = self.assistant_request_map[self.assistant_name]
        self.default_demonstrate_num = 2  # num of demonstrate example, during extending & enriching
        self.num_example_extend = 2  # triplet per new subtask, during extending
        self.default_extend_num = 5  # total new subtask num per request, during extending
        self.default_enrich_num = 50  # cumulative enrich num, during enriching
        self.num_example_enrich = 10  # num example per request, during enriching
        self.max_child_num = 12  # maximum children num
        self.max_example_num = 60000  # maximum example num per node
        self.min_extend_num = 3  # minimum request subtask num, during extending
        self.min_enrich_num = 5  # minimum request example num, during enriching
        self.default_extend_batch_size = 1  # prompt batch size, during extending
        self.default_enrich_batch_size = 1  # prompt batch size, during enriching

        self.max_retry_times = 20  # max retry times for online querying
        self.request_sleep_time = 20  # sleep time after online query fail
        self.attribute_of_interest = [self.unique_id, "reason", "examples", "config_filename", "raw_task_name"]
        self.attribute_meta_save = [self.unique_id, "reason", "config_filename", "raw_task_name"]
        self.general_outdir = getattr(self.root, "general_outdir",
                                      f"./mined_data_{getattr(self.root, self.unique_id)}")
        os.makedirs(self.general_outdir, exist_ok=True)
        self.bad_res_outfile = os.path.join(self.general_outdir, "bad_completions.jsonl")

    def gather_all_examples(self, ) -> Dict[str, List[Dict[str, str]]]:
        node_examples_map, total_count = dict(), 0
        for node in PreOrderIter(self.root):
            node_exampels = getattr(node, "examples", [])
            node_examples_map[getattr(node, self.unique_id)] = node_exampels
            total_count += len(node_exampels)
        node_examples_map["total_count"] = total_count
        return node_examples_map

    def prepare_prompt(self, ):
        raise NotImplementedError

    def prepare_tools(self):
        raise NotImplementedError

    def get_demonstrate_examples(self, node_to_prepare: AnyNode) -> List[Dict[str, str]]:
        demonstrate_node = node_to_prepare.parent if node_to_prepare.parent else node_to_prepare
        demonstrate_examples_pool = getattr(demonstrate_node, "examples", [])
        demonstrate_examples = random.sample(demonstrate_examples_pool, self.default_demonstrate_num) \
            if demonstrate_examples_pool else []
        return demonstrate_examples

    def add_node(self, node_info: Dict[str, Union[str, List[str]]]) -> AnyNode:
        assert self.unique_id in node_info and "parent" in node_info, \
            f"Invalid node to add: {node_info}: both .{self.unique_id} and .parent are required!"
        new_node = AnyNode(
            **node_info,
        )
        self.update_file(new_node, file_type="node", update_mode="new_file")
        self.name_to_node[getattr(new_node, self.unique_id)] = new_node
        logger.info(f"Node added: {node_info[self.unique_id]}")
        return new_node

    def add_node_to_tree(self, new_node: AnyNode):
        self.update_file(new_node, file_type="node", update_mode="new_file")
        self.name_to_node[getattr(new_node, self.unique_id)] = new_node
        logger.info(f"Node added: {getattr(new_node, self.unique_id)}")

    def update_file(
            self,
            node: AnyNode,
            update_mode: str = "new_file",
            file_type: str = "node",
    ):
        if update_mode == "new_file":
            node_save_file = os.path.join(self.general_outdir, node.config_filename)
            with open(node_save_file, "w", encoding="utf-8") as f_out:
                save_dict = {k: v for k, v in vars(node).items() if k in self.attribute_of_interest}
                json.dump(
                    save_dict,
                    f_out,
                    ensure_ascii=False,
                )
                f_out.write("\n")

    def handle_bad_completion(
            self,
            result: Dict[str, str],
            **kwargs,
    ):
        result.update(**kwargs)
        with open(self.bad_res_outfile, "a+", encoding="utf-8") as f_out:
            json.dump(
                result,
                f_out,
                ensure_ascii=False,
            )
            f_out.write("\n")

    def encode_prompt(
            self,
            base_prompt: str,
            demonstrate_examples: Optional[List[str]] = [],
            target_task: str = None,
            existing_children: List[AnyNode] = [],
            existing_siblings: List[AnyNode] = [],
            num_examples_per_time: int = None,
            extend_num: int = None,
            new_subtask: str = None,
            new_subtask_reason: str = None,
            target_children_num: int = None,
    ) -> str:
        raise NotImplementedError

    def formalize_taskname(self, taskname: str):
        raise NotImplementedError

    def save_to_local(self):
        """
        save meta file and each node info
        """
        # first check all node already saved to individual file
        for node in PreOrderIter(self.root):
            # config_file = getattr(node, "config_file") if getattr(node, "config_file", None)\
            #     else os.path.join(self.general_outdir, getattr(node, "config_filename"))
            config_file = os.path.join(self.general_outdir, getattr(node, "config_filename"))
            if not os.path.isfile(config_file):
                logger.warning(
                    f"Node info for {getattr(node, self.unique_id)} is not saved! Now saving to {config_file}")
                self.update_file(node, file_type="node", update_mode="new_file")

        # export meta file (set attriter in order to )
        exporter = DictExporter(
            attriter=lambda attrs: [(k, v) for k, v in attrs if k in self.attribute_meta_save]
        )
        meta_info = exporter.export(self.root)
        with open(os.path.join(self.general_outdir, meta_info_filename), "w", encoding="utf-8") as f_out:
            json.dump(
                meta_info,
                f_out,
                ensure_ascii=False,
            )
            f_out.write("\n")
        return

    def post_process_gpt3_response_enrich(
            self,
            response: str,
            current_new_subtask: str = None,
            current_new_subtask_reason: str = None,
    ) -> Tuple[Union[str, List[str]], Union[str, List[str]], List[Dict[str, str]]]:
        raise NotImplementedError

    @staticmethod
    def parse_node_config(
            node: AnyNode,
            config_field_name: str = "config_filename",
            **kwargs,
    ) -> AnyNode:
        config_filename = getattr(node, config_field_name, None)
        base_dir = kwargs.get("base_dir")
        config_file = os.path.join(base_dir, config_filename)
        if not config_filename or (not os.path.isfile(config_file)):
            logger.warning(f"Failed to load node info from {config_file}! Loading fail for {node}")
            return
        with open(config_file, encoding="utf-8") as f_in:
            task_config = json.load(f_in)
            for k, v in task_config.items():
                # if getattr(node, config_field_name, None):
                #     logger.warning(f"")
                setattr(node, k, v)

    def post_process_gpt3_response_extend(
            self,
            response: str
    ) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
        raise NotImplementedError

    @classmethod
    def from_tree_dict(
            cls,
            domain_tree_dict: Dict[str, Union[str, List[Dict[str, str]]]] = {},
            save_dir: str = None,
            out_dir: str = None,
            **kwargs,
    ):
        """
        Currently only for debugging use!
        """
        root_node = DictImporter().import_(domain_tree_dict)
        for node in PreOrderIter(root_node):
            cls.parse_node_config(node, base_dir=save_dir)
        if out_dir is not None:
            root_node.general_outdir = out_dir
        return cls(root_node, **kwargs)

    @classmethod
    def from_local_dir(
            cls,
            save_dir: str,
            out_dir: str = None,
            meta_file: str = meta_info_filename,
            **kwargs,
    ):
        # first load meta info, and infill each node info by loading config file

        with open(os.path.join(save_dir, meta_file)) as f_in:
            meta_info = json.load(f_in)
        root_node = DictImporter().import_(meta_info)
        for node in PreOrderIter(root_node):
            cls.parse_node_config(node, base_dir=save_dir)
        root_node.general_outdir = out_dir if out_dir is not None else save_dir
        return cls(root_node, **kwargs)


class EnDomainTreeRewrite(DomainTree):

    def encode_prompt(
            self,
            base_prompt: str,
            demonstrate_examples: Optional[List[str]] = [],
            target_task: str = None,
            existing_children: List[AnyNode] = [],
            existing_siblings: List[AnyNode] = [],
            num_examples_per_time: int = None,
            extend_num: int = None,
            new_subtask: str = None,
            new_subtask_reason: str = None,
            target_children_num: int = None,
    ) -> str:
        prompt = base_prompt
        prompt += f"\nTarget task: {target_task}\n"
        if len(demonstrate_examples) > 0:
            prompt += "Examples:\n"
            for idx, task_dict in enumerate(demonstrate_examples):
                (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
                instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
                input = "<noinput>" if input.lower() == "" else input
                prompt += "###\n"
                prompt += f"{idx + 1}. Instruction: {instruction}\n"
                prompt += f"Input: {input}\n"
                prompt += f"Output: {output}\n"
            prompt += "###\n"
        existing_children_names = [getattr(node, self.unique_id) for node in existing_children]
        existing_siblings_names = [getattr(node, self.unique_id) for node in existing_siblings]
        prompt += f"\nThe list of already existing subtasks for this target task is: {existing_children_names}.\n"
        prompt += f"The list of already existing peer tasks for this target task is: {existing_siblings_names}.\n"

        if target_children_num is not None:  # for extending
            prompt += f"\nThe target task should be decomposed into a total of {target_children_num} diverse and complementary subtasks, " \
                      f"and there are {len(existing_children)} existing subtasks. " \
                      f"Generate {extend_num} new subtasks with the corresponding reason, then list {num_examples_per_time} examples of this new subtask:"
        else:  # for enriching
            prompt += f"\nList {num_examples_per_time} examples of this new subtask below:"

        if new_subtask:  # for enriching
            prompt += "\n"
            prompt += f"\nNew subtask: {new_subtask}\n"
            prompt += f"Reason: {new_subtask_reason}"
        return prompt

    def prepare_prompt(self, ):
        self.extend_node_prompt = open("./prompt_bank/prompt_write_assistance_extend.txt").read() + "\n"
        self.enrich_node_prompt = open("./prompt_bank/prompt_write_assistance_enrich.txt").read() + "\n"

    def prepare_tools(self):
        logger.info("Preparint tools...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.openai_kwargs = {
            "model": "gpt-3.5-turbo",  # openai model type
            "temperature": 1.0,
            "top_p": 1.0,
            "n": 1,
            "logit_bias": {"50256": -100},  # prevent the <|endoftext|> token from being generated
        }
        if self.assistant_name == "claude":
            import anthropic
            self.claude_client = anthropic.Client(claude_key, proxy_url="http://127.0.0.1:2802")
            self.claude_kwargs = {
                "stop_sequences": [anthropic.HUMAN_PROMPT],
                "model": "claude-v1.3",  # anthropic model type
            }

    def formalize_taskname(self, taskname: str):
        taskname = re.sub('[^A-Za-z0-9]+', ' ', taskname)
        taskname = taskname.strip().replace(" ", "_").lower()
        # dedup
        if taskname in self.name_to_node:
            for i in range(10):
                dedup_taskname = f"{taskname}_{i}"
                if dedup_taskname not in self.name_to_node:
                    break

            taskname = dedup_taskname
        return taskname

    def post_process_gpt3_response_enrich(
            self,
            response: str,
            current_new_subtask: str = None,
            current_new_subtask_reason: str = None,
    ) -> Tuple[Union[str, List[str]], Union[str, List[str]], List[Dict[str, str]]]:
        if response is None:
            return None, current_new_subtask, current_new_subtask_reason
        stop_reason = response.get("stop_reason", "")
        raw_response = response.get("raw_response", "")

        raw_response = raw_response.replace("###", "").replace("### ", "").replace(" ###", "").replace(" ### ", "")
        new_subtask = current_new_subtask
        new_subtask_reason = current_new_subtask_reason
        new_subtask_examples = "\n" + raw_response.lstrip("Examples:")
        new_subtask_examples = re.split("\n+\d+\.\s+", new_subtask_examples)
        new_subtask_examples = new_subtask_examples[1:]
        instructions = []
        for idx, inst in enumerate(new_subtask_examples):
            # if the decoding stops due to length, the last example is likely truncated so we discard it
            if idx == len(new_subtask_examples) - 1 and stop_reason == "length":
                continue
            idx += 1
            splitted_data = re.split("Instruction:|Input:|Output:", inst)
            if len(splitted_data) != 4:
                continue
            inst = splitted_data[1].strip()
            input = splitted_data[2].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[3].strip().strip()
            # filter out too short or too long instructions
            if len(inst.split()) < 3 or len(inst.split()) > 150:
                continue
            # filter based on keywords that are not suitable for language models.
            blacklist = [
                "image",
                "images",
                "graph",
                "graphs",
                "picture",
                "pictures",
                "file",
                "files",
                "map",
                "maps",
                "draw",
                "plot",
                "go to",
                "video",
                "audio",
                "music",
                "flowchart",
                "diagram",
            ]
            blacklist += []
            if any(find_word_in_string(word, inst) for word in blacklist):
                continue
            # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
            # And it's a bit comfusing whether the model need to write a program or directly output the result.
            # Here we filter them out.
            # Note this is not a comprehensive filtering for all programming instructions.
            if inst.startswith("Write a program"):
                continue
            # filter those starting with punctuation
            if inst[0] in string.punctuation:
                continue
            # filter those starting with non-english character
            if not inst[0].isascii():
                continue
            # filter un-complete input
            if input.startswith("<") and input.endswith(">"):
                continue
            if input.startswith("(") and input.endswith(")"):
                continue
            instructions.append({"instruction": inst, "input": input, "output": output})
        return new_subtask, new_subtask_reason, instructions

    def post_process_gpt3_response_extend(
            self,
            response: str
    ) -> Tuple[List[str], List[str], List[List[Dict[str, str]]]]:
        stop_reason = response.get("stop_reason", "")
        raw_response = response.get("raw_response", "")

        raw_response = raw_response.replace("###", "").replace("### ", "").replace(" ###", "").replace(" ### ", "")
        raw_response = raw_response.replace("Example:", "Examples:")
        split_response = re.split("New subtask:|Reason:|Examples:", raw_response)
        split_response = split_response[1:]
        num_subtasks = len(split_response) // 3
        new_subtasks = []
        new_subtasks_reason = []
        new_subtasks_example = []
        for i in range(num_subtasks):
            new_subtask = split_response[i * 3].strip()
            new_subtask_reason = split_response[i * 3 + 1].strip()
            new_subtask_examples = split_response[i * 3 + 2]
            new_subtask_examples = re.split("\n+\d+\.\s+", new_subtask_examples)
            new_subtask_examples = new_subtask_examples[1:]
            instructions = []
            for idx, inst in enumerate(new_subtask_examples):
                # if the decoding stops due to length, the last example is likely truncated so we discard it
                if idx == len(new_subtask_examples) - 1 and stop_reason == "length":
                    continue
                splitted_data = re.split("Instruction:|Input:|Output:", inst)
                if len(splitted_data) != 4:
                    continue
                inst = splitted_data[1].strip()
                input = splitted_data[2].strip()
                input = "" if input.lower() == "<noinput>" else input
                output = splitted_data[3].strip().strip()
                # filter out too short or too long instructions
                if len(inst.split()) < 3 or len(inst.split()) > 150:
                    continue
                # filter based on keywords that are not suitable for language models.
                blacklist = [
                    "image",
                    "images",
                    "graph",
                    "graphs",
                    "picture",
                    "pictures",
                    "file",
                    "files",
                    "map",
                    "maps",
                    "draw",
                    "plot",
                    "go to",
                    "video",
                    "audio",
                    "music",
                    "flowchart",
                    "diagram",
                ]
                blacklist += []
                if any(find_word_in_string(word, inst) for word in blacklist):
                    continue
                # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
                # And it's a bit comfusing whether the model need to write a program or directly output the result.
                # Here we filter them out.
                # Note this is not a comprehensive filtering for all programming instructions.
                if inst.startswith("Write a program"):
                    continue
                # filter those starting with punctuation
                if inst[0] in string.punctuation:
                    continue
                # filter those starting with non-english character
                if not inst[0].isascii():
                    continue
                # filter un-complete input
                if input.startswith("<") and input.endswith(">"):
                    continue
                if input.startswith("(") and input.endswith(")"):
                    continue
                instructions.append({"instruction": inst, "input": input, "output": output})
            new_subtasks.append(new_subtask)
            new_subtasks_reason.append(new_subtask_reason)
            new_subtasks_example.append(instructions)
        return new_subtasks, new_subtasks_reason, new_subtasks_example


class EnDomainTreeBrainstorming(DomainTree):

    def encode_prompt(
            self,
            base_prompt: str,
            demonstrate_examples: Optional[List[str]] = [],
            target_task: str = None,
            existing_children: List[AnyNode] = [],
            existing_siblings: List[AnyNode] = [],
            num_examples_per_time: int = None,
            extend_num: int = None,
            new_subtask: str = None,
            new_subtask_reason: str = None,
            target_children_num: int = None,
    ) -> str:
        prompt = base_prompt
        prompt += f"\nTarget task: {target_task}\n"
        if len(demonstrate_examples) > 0:
            prompt += "Examples:\n"
            for idx, task_dict in enumerate(demonstrate_examples):
                (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
                instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
                input = "<noinput>" if input.lower() == "" else input
                prompt += "###\n"
                prompt += f"Instruction: {instruction}\n"  # without index
                prompt += f"Input: {input}\n"
                prompt += f"Output: {output}\n"
            prompt += "###\n"
        existing_children_names = [getattr(node, self.unique_id) for node in existing_children]
        existing_siblings_names = [getattr(node, self.unique_id) for node in existing_siblings]
        prompt += f"\nThe list of already existing subtasks for this target task is: {existing_children_names}.\n"
        prompt += f"The list of already existing peer tasks for this target task is: {existing_siblings_names}.\n"

        if target_children_num is not None:  # for extending
            prompt += f"\nThe target task should be decomposed into a total of {target_children_num} diverse and complementary subtasks, " \
                      f"and there are {len(existing_children)} existing subtasks. " \
                      f"Generate {extend_num} new subtasks with the corresponding reason, then list {num_examples_per_time} examples of this new subtask:"
        else:  # for enriching
            prompt += f"\nList {num_examples_per_time} examples of this new subtask below:"

        if new_subtask:  # for enriching
            prompt += "\n"
            prompt += f"\nNew subtask: {new_subtask}\n"
            prompt += f"Reason: {new_subtask_reason}"
        return prompt

    def prepare_prompt(self, ):
        self.extend_node_prompt = open("./prompt_bank/prompt_brainstorming_extend.txt").read() + "\n"
        self.enrich_node_prompt = open("./prompt_bank/prompt_brainstorming_enrich.txt").read() + "\n"

    def prepare_tools(self):
        logger.info("Preparint tools...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.openai_kwargs = {
            "model": "gpt-3.5-turbo",  # openai model type
            "temperature": 1.0,
            "top_p": 1.0,
            "n": 1,
            "logit_bias": {"50256": -100},  # prevent the <|endoftext|> token from being generated
        }
        if self.assistant_name == "claude":
            import anthropic
            self.claude_client = anthropic.Client(claude_key, proxy_url="http://127.0.0.1:2802")
            self.claude_kwargs = {
                "stop_sequences": [anthropic.HUMAN_PROMPT],
                "model": "claude-v1.3",  # anthropic model type
            }

    def formalize_taskname(self, taskname: str):
        taskname = re.sub('[^A-Za-z0-9]+', ' ', taskname)
        taskname = taskname.strip().replace(" ", "_").lower()
        # dedup
        if taskname in self.name_to_node:
            for i in range(10):
                dedup_taskname = f"{taskname}_{i}"
                if dedup_taskname not in self.name_to_node:
                    break

            taskname = dedup_taskname
        return taskname

    def post_process_gpt3_response_enrich(
            self,
            response: str,
            current_new_subtask: str = None,
            current_new_subtask_reason: str = None,
    ) -> Tuple[Union[str, List[str]], Union[str, List[str]], List[Dict[str, str]]]:
        if response is None:
            return None, current_new_subtask, current_new_subtask_reason
        stop_reason = response.get("stop_reason", "")
        raw_response = response.get("raw_response", "")

        raw_response = raw_response.replace("###", "").replace("### ", "").replace(" ###", "").replace(" ### ", "")
        new_subtask = current_new_subtask
        new_subtask_reason = current_new_subtask_reason
        new_subtask_examples = "\n" + raw_response.lstrip("Examples:")
        new_subtask_examples = re.split("Instruction:", new_subtask_examples)
        new_subtask_examples = new_subtask_examples[1:]
        instructions = []
        for idx, inst in enumerate(new_subtask_examples):
            # if the decoding stops due to length, the last example is likely truncated so we discard it
            if idx == len(new_subtask_examples) - 1 and stop_reason == "length":
                continue
            idx += 1
            splitted_data = re.split("Instruction:|Input:|Output:", inst)
            if len(splitted_data) != 3:
                continue
            inst = splitted_data[0].strip()
            input = splitted_data[1].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[2].strip().strip()
            # filter out too short or too long instructions
            if len(inst.split()) < 3 or len(inst.split()) > 150:
                continue
            # filter based on keywords that are not suitable for language models.
            blacklist = [
                "image",
                "images",
                "graph",
                "graphs",
                "picture",
                "pictures",
                "file",
                "files",
                # "map",
                # "maps",
                "draw",
                "plot",
                "go to",
                "video",
                "audio",
                "music",
                "flowchart",
                "diagram",
            ]
            blacklist += []
            if any(find_word_in_string(word, inst) for word in blacklist):
                continue
            # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
            # And it's a bit comfusing whether the model need to write a program or directly output the result.
            # Here we filter them out.
            # Note this is not a comprehensive filtering for all programming instructions.
            if inst.startswith("Write a program"):
                continue
            # filter those starting with punctuation
            if inst[0] in string.punctuation:
                continue
            # filter those starting with non-english character
            if not inst[0].isascii():
                continue
            # filter un-complete input
            if input.startswith("<") and input.endswith(">"):
                continue
            if input.startswith("(") and input.endswith(")"):
                continue
            instructions.append({"instruction": inst, "input": input, "output": output})
        return new_subtask, new_subtask_reason, instructions

    def post_process_gpt3_response_extend(
            self,
            response: str
    ) -> Tuple[List[str], List[str], List[List[Dict[str, str]]]]:
        stop_reason = response.get("stop_reason", "")
        raw_response = response.get("raw_response", "")

        raw_response = raw_response.replace("###", "").replace("### ", "").replace(" ###", "").replace(" ### ", "")
        raw_response = raw_response.replace("Example:", "Examples:")
        split_response = re.split("New subtask:|Reason:|Examples:", raw_response)
        split_response = split_response[1:]
        num_subtasks = len(split_response) // 3
        new_subtasks = []
        new_subtasks_reason = []
        new_subtasks_example = []
        for i in range(num_subtasks):
            new_subtask = split_response[i * 3].strip()
            new_subtask_reason = split_response[i * 3 + 1].strip()
            new_subtask_examples = split_response[i * 3 + 2]
            new_subtask_examples = re.split("Instruction:", new_subtask_examples)
            new_subtask_examples = new_subtask_examples[1:]
            instructions = []
            for idx, inst in enumerate(new_subtask_examples):
                # if the decoding stops due to length, the last example is likely truncated so we discard it
                if idx == len(new_subtask_examples) - 1 and stop_reason == "length":
                    continue
                splitted_data = re.split("Instruction:|Input:|Output:", inst)
                if len(splitted_data) != 3:
                    continue
                inst = splitted_data[0].strip()
                input = splitted_data[1].strip()
                input = "" if input.lower() == "<noinput>" else input
                output = splitted_data[2].strip().strip()
                # filter out too short or too long instructions
                if len(inst.split()) < 3 or len(inst.split()) > 150:
                    continue
                # filter based on keywords that are not suitable for language models.
                blacklist = [
                    "image",
                    "images",
                    "graph",
                    "graphs",
                    "picture",
                    "pictures",
                    "file",
                    "files",
                    # "map",
                    # "maps",
                    "draw",
                    "plot",
                    "go to",
                    "video",
                    "audio",
                    "music",
                    "flowchart",
                    "diagram",
                ]
                blacklist += []
                if any(find_word_in_string(word, inst) for word in blacklist):
                    continue
                # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
                # And it's a bit comfusing whether the model need to write a program or directly output the result.
                # Here we filter them out.
                # Note this is not a comprehensive filtering for all programming instructions.
                if inst.startswith("Write a program"):
                    continue
                # filter those starting with punctuation
                if inst[0] in string.punctuation:
                    continue
                # filter those starting with non-english character
                if not inst[0].isascii():
                    continue
                # filter un-complete input
                if input.startswith("<") and input.endswith(">"):
                    continue
                if input.startswith("(") and input.endswith(")"):
                    continue
                instructions.append({"instruction": inst, "input": input, "output": output})
            new_subtasks.append(new_subtask)
            new_subtasks_reason.append(new_subtask_reason)
            new_subtasks_example.append(instructions)
        return new_subtasks, new_subtasks_reason, new_subtasks_example


class EnDomainTreeMath(DomainTree):

    def encode_prompt(
            self,
            base_prompt: str,
            demonstrate_examples: Optional[List[str]] = [],
            target_task: str = None,
            existing_children: List[AnyNode] = [],
            existing_siblings: List[AnyNode] = [],
            num_examples_per_time: int = None,
            extend_num: int = None,
            new_subtask: str = None,
            new_subtask_reason: str = None,
            target_children_num: int = None,
    ) -> str:
        prompt = base_prompt
        prompt += f"\nTarget task: {target_task}\n"
        if len(demonstrate_examples) > 0:
            prompt += "Examples:\n"
            for idx, task_dict in enumerate(demonstrate_examples):
                (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
                instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
                input = "<noinput>" if input.lower() == "" else input
                prompt += "###\n"
                prompt += f"Instruction: {instruction}\n"  # without index
                prompt += f"Input: {input}\n"
                prompt += f"Output: {output}\n"
            prompt += "###\n"
        existing_children_names = [getattr(node, self.unique_id) for node in existing_children]
        existing_siblings_names = [getattr(node, self.unique_id) for node in existing_siblings]
        prompt += f"\nThe list of already existing subtasks for this target task is: {existing_children_names}.\n"
        prompt += f"The list of already existing peer tasks for this target task is: {existing_siblings_names}.\n"

        if target_children_num is not None:  # for extending
            prompt += f"\nThe target task should be decomposed into a total of {target_children_num} diverse and complementary subtasks, " \
                      f"and there are {len(existing_children)} existing subtasks. " \
                      f"Generate {extend_num} new subtasks with the corresponding reason, then list {num_examples_per_time} examples of this new subtask:"
        else:  # for enriching
            prompt += f"\nList {num_examples_per_time} examples of this new subtask below:"

        if new_subtask:  # for enriching
            prompt += "\n"
            prompt += f"\nNew subtask: {new_subtask}\n"
            prompt += f"Reason: {new_subtask_reason}"
        return prompt

    def prepare_prompt(self, ):
        self.extend_node_prompt = open("./prompt_bank/prompt_math_extend.txt").read() + "\n"
        self.enrich_node_prompt = open("./prompt_bank/prompt_math_enrich.txt").read() + "\n"

    def prepare_tools(self):
        logger.info("Preparint tools...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.openai_kwargs = {
            "model": "gpt-3.5-turbo",  # openai model type
            "temperature": 1.0,
            "top_p": 1.0,
            "n": 1,
            "logit_bias": {"50256": -100},  # prevent the <|endoftext|> token from being generated
        }
        if self.assistant_name == "claude":
            import anthropic
            self.claude_client = anthropic.Client(claude_key, proxy_url="http://127.0.0.1:2802")
            self.claude_kwargs = {
                "stop_sequences": [anthropic.HUMAN_PROMPT],
                "model": "claude-v1.3",  # anthropic model type
            }

    def formalize_taskname(self, taskname: str):
        taskname = re.sub('[^A-Za-z0-9]+', ' ', taskname)
        taskname = taskname.strip().replace(" ", "_").lower()
        # dedup
        if taskname in self.name_to_node:
            for i in range(10):
                dedup_taskname = f"{taskname}_{i}"
                if dedup_taskname not in self.name_to_node:
                    break

            taskname = dedup_taskname
        return taskname

    def post_process_gpt3_response_enrich(
            self,
            response: str,
            current_new_subtask: str = None,
            current_new_subtask_reason: str = None,
    ) -> Tuple[Union[str, List[str]], Union[str, List[str]], List[Dict[str, str]]]:
        if response is None:
            return None, current_new_subtask, current_new_subtask_reason
        stop_reason = response.get("stop_reason", "")
        raw_response = response.get("raw_response", "")

        raw_response = raw_response.replace("###", "").replace("### ", "").replace(" ###", "").replace(" ### ", "")
        new_subtask = current_new_subtask
        new_subtask_reason = current_new_subtask_reason
        new_subtask_examples = "\n" + raw_response.lstrip("Examples:")
        new_subtask_examples = re.split("Instruction:", new_subtask_examples)
        new_subtask_examples = new_subtask_examples[1:]
        instructions = []
        for idx, inst in enumerate(new_subtask_examples):
            # if the decoding stops due to length, the last example is likely truncated so we discard it
            if idx == len(new_subtask_examples) - 1 and stop_reason == "length":
                continue
            idx += 1
            splitted_data = re.split("Instruction:|Input:|Output:", inst)
            if len(splitted_data) != 3:
                continue
            inst = splitted_data[0].strip()
            input = splitted_data[1].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[2].strip().strip()
            if "Answer: " in output and "\\boxed" not in output[output.index("Answer: "):]:
                answer_index = output.index("Answer: ")
                answer = output[answer_index:].lstrip("Answer: ")
                answer = answer[0] + "\\boxed{" + answer[1:-1] + "}" + answer[-1]
                output = output[:answer_index] + "Answer: " + answer
            # filter based on keywords that are not suitable for language models.
            blacklist = [
                "image",
                "images",
                # "graph",
                # "graphs",
                "picture",
                "pictures",
                "file",
                "files",
                # "map",
                # "maps",
                "draw",
                "plot",
                "go to",
                "video",
                "audio",
                "music",
                "flowchart",
                "diagram",
            ]
            blacklist += []
            if any(find_word_in_string(word, inst) for word in blacklist):
                continue
            # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
            # And it's a bit comfusing whether the model need to write a program or directly output the result.
            # Here we filter them out.
            # Note this is not a comprehensive filtering for all programming instructions.
            if inst.startswith("Write a program"):
                continue
            # filter those starting with punctuation
            if inst[0] in string.punctuation:
                continue
            # filter those starting with non-english character
            if not inst[0].isascii():
                continue
            # filter un-complete input
            if input.startswith("<") and input.endswith(">"):
                continue
            if input.startswith("(") and input.endswith(")"):
                continue
            instructions.append({"instruction": inst, "input": input, "output": output})
        return new_subtask, new_subtask_reason, instructions

    def post_process_gpt3_response_extend(
            self,
            response: str
    ) -> Tuple[List[str], List[str], List[List[Dict[str, str]]]]:
        stop_reason = response.get("stop_reason", "")
        raw_response = response.get("raw_response", "")

        raw_response = raw_response.replace("###", "").replace("### ", "").replace(" ###", "").replace(" ### ", "")
        raw_response = raw_response.replace("Example:", "Examples:")
        split_response = re.split("New subtask:|Reason:|Examples:", raw_response)
        split_response = split_response[1:]
        num_subtasks = len(split_response) // 3
        new_subtasks = []
        new_subtasks_reason = []
        new_subtasks_example = []
        for i in range(num_subtasks):
            new_subtask = split_response[i * 3].strip()
            new_subtask_reason = split_response[i * 3 + 1].strip()
            new_subtask_examples = split_response[i * 3 + 2]
            new_subtask_examples = re.split("Instruction:", new_subtask_examples)
            new_subtask_examples = new_subtask_examples[1:]
            instructions = []
            for idx, inst in enumerate(new_subtask_examples):
                # if the decoding stops due to length, the last example is likely truncated so we discard it
                if idx == len(new_subtask_examples) - 1 and stop_reason == "length":
                    continue
                splitted_data = re.split("Instruction:|Input:|Output:", inst)
                if len(splitted_data) != 3:
                    continue
                inst = splitted_data[0].strip()
                input = splitted_data[1].strip()
                input = "" if input.lower() == "<noinput>" else input
                output = splitted_data[2].strip().strip()
                if "Answer: " in output and "\\boxed" not in output[output.index("Answer: "):]:
                    answer_index = output.index("Answer: ")
                    answer = output[answer_index:].lstrip("Answer: ")
                    answer = answer[0] + "\\boxed{" + answer[1:-1] + "}" + answer[-1]
                    output = output[:answer_index] + "Answer: " + answer
                # filter based on keywords that are not suitable for language models.
                blacklist = [
                    "image",
                    "images",
                    # "graph",
                    # "graphs",
                    "picture",
                    "pictures",
                    "file",
                    "files",
                    # "map",
                    # "maps",
                    "draw",
                    "plot",
                    "go to",
                    "video",
                    "audio",
                    "music",
                    "flowchart",
                    "diagram",
                ]
                blacklist += []
                if any(find_word_in_string(word, inst) for word in blacklist):
                    continue
                # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
                # And it's a bit comfusing whether the model need to write a program or directly output the result.
                # Here we filter them out.
                # Note this is not a comprehensive filtering for all programming instructions.
                if inst.startswith("Write a program"):
                    continue
                # filter those starting with punctuation
                if inst[0] in string.punctuation:
                    continue
                # filter those starting with non-english character
                if not inst[0].isascii():
                    continue
                # filter un-complete input
                if input.startswith("<") and input.endswith(">"):
                    continue
                if input.startswith("(") and input.endswith(")"):
                    continue
                instructions.append({"instruction": inst, "input": input, "output": output})
            new_subtasks.append(new_subtask)
            new_subtasks_reason.append(new_subtask_reason)
            new_subtasks_example.append(instructions)
        return new_subtasks, new_subtasks_reason, new_subtasks_example


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def test_extending(
        domain_tree: DomainTree,
        max_depth: int = 2,
        extend_nums: List = None,
        extend_batch_size: int = None
):
    queue = [_ for _ in PreOrderIter(domain_tree.root)]
    while len(queue) > 0:
        node = queue.pop(0)
        logger.info(f"Processing {node.task_name}, depth: {node.depth}")
        if node.depth >= max_depth:
            continue
        new_nodes: List[AnyNode] = domain_tree.extend_node_children(
            node,
            extend_num=extend_nums[node.depth],
            extend_batch_size=extend_batch_size,
        )
        domain_tree.save_to_local()
        for new_node in new_nodes:
            if new_node.depth < max_depth:
                queue.append(new_node)
    domain_tree.save_to_local()
    return


def test_enriching(
        domain_tree: DomainTree,
        enrich_nums: List = None,
        enrich_batch_size: int = None,
):
    queue = [_ for _ in PreOrderIter(domain_tree.root)]
    for node in queue:
        logger.info(f"Processing {node.task_name}, depth: {node.depth}")
        domain_tree.enrich_node_samples(
            node,
            enrich_num=enrich_nums[node.depth],
            enrich_batch_size=enrich_batch_size,
        )
        domain_tree.save_to_local()
    return


def test_prune(
        domain_tree: DomainTree,
        prune_threshold: float,
        num_cpus=1,
):
    queue = [_ for _ in LevelOrderIter(domain_tree.root)]
    root = queue[0]
    queue = queue[1:]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    all_subtasks = [root.raw_task_name]
    all_subtask_tokens = [scorer._tokenizer.tokenize(subtask_name) for subtask_name in all_subtasks]
    pruned_subtask_name = []
    for node in tqdm.tqdm(queue):
        new_subtask_tokens = scorer._tokenizer.tokenize(node.raw_task_name)
        with Pool(num_cpus) as p:
            rouge_scores = p.map(
                partial(rouge_scorer._score_lcs, new_subtask_tokens),
                all_subtask_tokens,
            )
        rouge_scores = [score.fmeasure for score in rouge_scores]
        # most_similar_instructions = {
        #     all_subtasks[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
        # }
        if max(rouge_scores) > prune_threshold:  # pruning this subtask
            pruned_subtask_name.append(node.task_name)
            continue
        all_subtasks.append(node.raw_task_name)
        all_subtask_tokens.append(new_subtask_tokens)
    return pruned_subtask_name


def test_filter(
        domain_tree: DomainTree,
        filter_threshold: float,
        num_cpus=64,
):
    queue = [_ for _ in LevelOrderIter(domain_tree.root)]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    logger.info("Start filtering.")
    for node in tqdm.tqdm(queue):
        keep = 0
        all_instruction_tokens = []
        all_examples = []
        for instruction_data_entry in tqdm.tqdm(node.examples):
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            # most_similar_instructions = {
            #     all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            # }
            if len(rouge_scores) != 0 and max(rouge_scores) > filter_threshold:
                continue
            else:
                keep += 1
            all_examples.append(instruction_data_entry)
            # all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
        node.examples = all_examples
        logger.info(f"Subtask: {node.task_name}. Filtered instances for this subtask: {keep}")
        domain_tree.save_to_local()
        domain_tree.update_file(node)
    return


def test_sample(domain_tree: DomainTree,
                export_file: str,
                sample_example_num: int,
                sample_max_depth: int,
                sample_use_pruned: bool,
                pruned_subtasks_name: List[str]
                ):
    queue = [_ for _ in LevelOrderIter(domain_tree.root) if _.depth <= sample_max_depth]
    if sample_use_pruned:
        pruned_subtasks_name = [_.task_name for _ in queue if _.task_name in pruned_subtasks_name]
    else:
        pruned_subtasks_name = []
    logger.info(f"All subtasks: {len(queue)}. Pruned subtasks: {len(pruned_subtasks_name)}")
    # sample_example_per_node = sample_example_num // (len(queue) - len(pruned_subtasks_name)) + 1
    # logger.info(f"Sample examples per node: {sample_example_per_node}")
    data = []
    all_examples_num = 0
    for node in queue:
        all_examples_num += min(len(node.examples), 500)
    for node in queue:
        sample_example_per_node = min(int(sample_example_num * (min(len(node.examples), 500) / all_examples_num)) + 1,
                                      len(node.examples))
        logger.info(f"Sample examples num for {node.task_name}: {sample_example_per_node}")
        data += random.sample(node.examples, k=sample_example_per_node)
    data = data[:sample_example_num]
    if len(data) < sample_example_num:
        logger.info(f"Have no enough examples, sample {sample_example_num - len(data)} examples from root.")
        data += random.sample(queue[0].examples, k=sample_example_num - len(data))
    with open(export_file, "w") as fout:
        json.dump(data, fout, indent=4)
    logger.info(f"All sampled examples {len(data)}")


def run_extend(args):
    save_dir = args.save_dir
    out_dir = args.out_dir
    assistant_name = args.assistant_name
    extend_nums = [int(_) for _ in args.extend_nums.split(",")]
    max_depth = args.max_depth
    extend_batch_size = args.extend_batch_size
    TreeFactory = args.tree_map[args.lang][args.domain]
    domain_tree: DomainTree = TreeFactory.from_local_dir(
        save_dir=save_dir,
        out_dir=out_dir,
        assistant_name=assistant_name,
    )
    test_extending(domain_tree,
                   max_depth=max_depth,
                   extend_nums=extend_nums,
                   extend_batch_size=extend_batch_size)
    return


def run_enrich(args):
    save_dir = args.save_dir
    out_dir = args.out_dir
    assistant_name = args.assistant_name
    enrich_nums = [int(_) for _ in args.enrich_nums.split(",")]
    enrich_batch_size = args.enrich_batch_size
    TreeFactory = args.tree_map[args.lang][args.domain]
    domain_tree: DomainTree = TreeFactory.from_local_dir(
        save_dir=save_dir,
        out_dir=out_dir,
        assistant_name=assistant_name,
    )
    test_enriching(
        domain_tree,
        enrich_nums=enrich_nums,
        enrich_batch_size=enrich_batch_size
    )
    return


def run_prune(args):
    assistant_name = args.assistant_name
    save_dir = args.save_dir
    out_dir = args.out_dir
    pruned_file = args.pruned_file
    TreeFactory = args.tree_map[args.lang][args.domain]
    domain_tree: DomainTree = TreeFactory.from_local_dir(save_dir=save_dir, out_dir=out_dir,
                                                         assistant_name=assistant_name)
    pruned_subtask_name = test_prune(domain_tree, args.prune_threshold)
    with open(pruned_file, "w") as fout:
        json.dump(pruned_subtask_name, fout)
    domain_tree.save_to_local()


def run_filter(args):
    assistant_name = args.assistant_name
    save_dir = args.save_dir
    out_dir = args.out_dir
    filter_threshold = args.filter_threshold
    TreeFactory = args.tree_map[args.lang][args.domain]
    domain_tree: DomainTree = TreeFactory.from_local_dir(save_dir=save_dir, out_dir=out_dir,
                                                         assistant_name=assistant_name)
    test_filter(domain_tree, filter_threshold)
    shutil.copy(args.pruned_file, os.path.join(out_dir, "pruned_subtasks_name.json"))


def run_sample(args):
    assistant_name = args.assistant_name
    save_dir = args.save_dir
    export_file = args.export_file
    sample_example_num = args.sample_example_num
    sample_max_depth = args.sample_max_depth
    sample_use_pruned = args.sample_use_pruned
    pruned_subtasks_name = json.loads(open(args.pruned_file, "r").read())
    TreeFactory = args.tree_map[args.lang][args.domain]
    domain_tree: DomainTree = TreeFactory.from_local_dir(save_dir=save_dir, out_dir=save_dir + "_tmp",
                                                         assistant_name=assistant_name)
    test_sample(domain_tree, export_file=export_file, sample_example_num=sample_example_num,
                sample_max_depth=sample_max_depth, sample_use_pruned=sample_use_pruned,
                pruned_subtasks_name=pruned_subtasks_name)
    os.rmdir(save_dir + "_tmp")


func_action_mapping = {
    "extend": run_extend,
    "enrich": run_enrich,
    "prune": run_prune,
    "filter": run_filter,
    "sample": run_sample,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, required=True, help='action to do')
    parser.add_argument('--save_dir', type=str, help='original tree dir')
    parser.add_argument('--out_dir', type=str, help='dir to save tree after operations (recommended to use a new one instead of reusing save_dir)')
    parser.add_argument('--export_file', type=str, help='single json file to store all exported examples')
    parser.add_argument('--lang', type=str, default="en", help='either zh or en')
    parser.add_argument('--domain', type=str, default="rewrite", help='target domain')
    parser.add_argument('--extend_nums', type=str, help='children num to add during extending')
    parser.add_argument('--extend_batch_size', type=int, default=None, help='prompt batch size for extending')
    parser.add_argument('--max_depth', type=int, help='extend max depth')
    parser.add_argument('--enrich_nums', type=str, help='examples to add during enriching')
    parser.add_argument('--enrich_batch_size', type=int, default=None, help='prompt batch size for enriching')
    parser.add_argument('--prune_threshold', type=float, default=0.7, help='threshold for sub-task pruning')
    parser.add_argument('--pruned_file', type=str, default=None, help='file to store pruned subtasks name list')
    parser.add_argument('--filter_threshold', type=float, default=0.7, help='threshold for filter examples')
    parser.add_argument('--sample_example_num', type=int, default=50000, help='data num to use during sampling')
    parser.add_argument('--sample_max_depth', type=int, default=3, help='max depth for sampling')
    parser.add_argument('--sample_use_pruned', action="store_true", help='use pruned for sampling')
    parser.add_argument('--assistant_name', type=str, help='using either openai or claude')
    args = parser.parse_args()
    args.tree_map = {"en": {"rewrite": EnDomainTreeRewrite,
                            "brainstorming": EnDomainTreeBrainstorming,
                            "math": EnDomainTreeMath,
                            }}
    random.seed(42)
    print(args)
    # further add sanity check
    func_action_mapping[args.action](args)


if __name__ == "__main__":
    main()
