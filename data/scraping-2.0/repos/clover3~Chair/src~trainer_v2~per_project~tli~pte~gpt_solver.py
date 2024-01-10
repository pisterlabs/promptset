import json
import os
from collections import Counter

from cpath import output_path
from trainer_v2.chair_logging import c_log
from utils.open_ai_api import OpenAIProxy
from dataset_specific.scientsbank.pte_solver_if import PTESolverIF
from dataset_specific.scientsbank.pte_data_types import ReferenceAnswer, Facet
from typing import Dict, List

from iter_util import load_jsonl
from misc_lib import path_join
from utils.open_ai_api import parse_instruct_gpt_response, get_parse_gpt_response_fn

template_single_facet = """

Student answer: {}.
Reference answer: {}.
Facet: ({}, {})

The facet is a relation extracted from the reference answer. 
In the example above, does the student answer entail the given facet? 
Answer with Yes/No

"""


class ResponseCacher:
    def __init__(self, save_path):
        self.log_path = save_path
        self.log_file = None

    def write(self, e):
        if self.log_file is None:
            self.log_file = open(self.log_path, "a")

        self.log_file.write(json.dumps(e) + "\n")
        self.log_file.flush()

    def read_caches_as_d(self):
        out_d = {}
        if os.path.exists(self.log_path):
            j_list = load_jsonl(self.log_path)
            for j in j_list:
                key = get_key_for_pte_j_entry(j)
                out_d[key] = j
        c_log.info("%d items parsed", len(out_d))
        return out_d

    def read_caches(self) -> List[dict]:
        j_list = load_jsonl(self.log_path)
        return j_list


class GPTRequesterForPTE(PTESolverIF):
    def __init__(self,
                 open_ai_proxy: OpenAIProxy,
                 prompt_template,
                 cacher: ResponseCacher,
                 ):
        self.tli_cache = {}
        self.prompt_template = prompt_template
        self.proxy: OpenAIProxy = open_ai_proxy
        self.cacher: ResponseCacher = cacher
        self.cache_d = self.cacher.read_caches_as_d()
        self.cache_hit = 0
        self.n_seen = 0

    def solve(self,
              reference_answer: ReferenceAnswer,
              student_answer: str,
              facet: Facet) -> float:
        self.n_seen += 1
        premise_like = student_answer
        hypothesis_like = reference_answer.text
        key = "{}_{}_{}".format(
            reference_answer.id,
            student_answer,
            facet.id,
        )
        if key in self.cache_d:
            self.cache_hit += 1
        else:
            if self.cache_hit:
                c_log.info("%d records from cache", self.cache_hit)
                self.cache_hit = 0

            prompt = self.prompt_template.format(premise_like, hypothesis_like, facet.govText, facet.modText)
            c_log.debug("Issue request")
            c_log.debug(prompt)
            response = self.proxy.request(prompt)
            j_save = {
                'reference_answer.id': reference_answer.id,
                'student_answer': student_answer,
                'facet.id': facet.id,
                'response': response
            }
            c_log.debug("Received")
            self.cacher.write(j_save)
            self.cache_d[key] = j_save
        output_score = 0
        return float(output_score)


class ResponseTextParser:
    def __init__(self):
        self.irregular = Counter()

    def parse_response(self, text):
        if "Yes" in text:
            decision = True
        elif "No" in text:
            decision = False
        elif "Partial" in text:
            decision = False
        else:
            raise ValueError(text)

        if len(text.strip()) > 4:
            self.irregular[text] += 1
        return decision

    def end(self):
        if len(self.irregular) > 0:
            print(self.irregular)


def get_key_for_pte_j_entry(j):
    key = "{}_{}_{}".format(
        j['reference_answer.id'],
        j['student_answer'],
        j['facet.id'],
    )
    return key


class GPTSolverForPTE(PTESolverIF):
    def __init__(self,
                 parse_gpt_response_fn,
                 cacher: ResponseCacher,
                 name: str
                 ):
        self.cacher = cacher
        j_list = cacher.read_caches()
        self.parse_gpt_response = parse_gpt_response_fn
        self.decision_d = self.parse_solve(j_list)
        c_log.info("%d keys loaded", len(self.decision_d))
        self.name = name

    def get_name(self):
        return self.name

    def parse_solve(self, json_list) -> Dict[str, bool]:
        text_parser = ResponseTextParser()
        decision_d = {}
        for j in json_list:
            key = get_key_for_pte_j_entry(j)
            text = self.parse_gpt_response(j['response'])
            decision = text_parser.parse_response(text)
            decision_d[key] = decision
        text_parser.end()
        return decision_d

    def solve(self,
              reference_answer: ReferenceAnswer,
              student_answer: str,
              facet: Facet) -> float:
        key = "{}_{}_{}".format(reference_answer.id, student_answer, facet.id)
        decision = self.decision_d[key]
        if decision:
            score = 1
        else:
            score = 0
        return float(score)


def get_log_save_path(engine, split):
    log_path = path_join(output_path, "pte_scientsbank", "gpt", f"{engine}_{split}.json")
    return log_path


def get_gpt_requester(engine, split):
    proxy = OpenAIProxy(engine)
    log_path = get_log_save_path(engine, split)
    cacher = ResponseCacher(log_path)
    return GPTRequesterForPTE(proxy, template_single_facet, cacher)


def get_gpt_read_solver(engine: str, split: str):
    log_path = get_log_save_path(engine, split)
    cacher = ResponseCacher(log_path)
    parse_response_fn = get_parse_gpt_response_fn(engine)
    return GPTSolverForPTE(parse_response_fn, cacher, engine)


def main():
    sample2 = """
Student answer: By letting it sit in a dish for a day.
Reference answer: The water was evaporated, leaving the salt.
Facets: (evaporated, water), (leaving, evaporated), (leaving, salt)

The facets above which are represented as pair of words, are the relations extracted from the reference answer. 
In this example does the student answer entails each of the facets? 
Answer with Yes or No, separated by comma(,).

"""


if __name__ == "__main__":
    main()
