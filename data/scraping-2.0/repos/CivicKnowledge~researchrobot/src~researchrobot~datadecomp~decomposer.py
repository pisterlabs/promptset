"""The Decomposer makes requests to GPT-3 to decompose questions"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import openai

import researchrobot.datadecomp.prompts as prmpt


@dataclass
class Decomposition:
    entity: str
    measures: list
    factors: dict
    dimensions: list

    question: str = None
    description: str = None

    def to_json(self):
        return json.dumps(self.__dict__, indent=4)

    @classmethod
    def from_json(cls, json_str):

        if isinstance(json_str, str):
            d = json.loads(json_str)
        else:
            d = json_str

        # Lowercase all of the keys in d
        d = {k.lower(): v for k, v in d.items()}

        if isinstance(d["entity"], list):
            d["entity"] = d["entity"][0]

        return cls(**d)

    def as_list(self):
        """Fatten the decomposition and return as a flat list"""
        return [self.entity] + self.measures + self.factors + self.dimensions

    def as_text(self):
        pass

    def as_dict(self):
        return self.__dict__

    def __str__(self):
        return self.to_json()


class Decomposer:

    decompose_prompt = "decompose_steps.txt"
    default_model = "gpt-4"

    def __init__(self, model_id=None):

        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.stop = None

        self.model_id = model_id if model_id is not None else self.default_model

    def _is_multi(self, q):
        return isinstance(q, (list, tuple))

    def make_question_prompt(self, q):

        q_indent = " " * 4
        plural = self._is_multi(q)

        q = q_indent + str(q) if not plural else f"\n{q_indent}".join(str(e) for e in q)

        pre = Path(prmpt.__file__).parent.joinpath(self.decompose_prompt).read_text()

        # Examples and intro is are ued in decompose.txt, not decompose_steps.txt
        examples_fn = (
            "decompose_example_multi.txt" if plural else "decompose_example_single.txt"
        )
        examples = Path(prmpt.__file__).parent.joinpath(examples_fn).read_text()
        intro = (
            f"What is the JSON formatted decomposition for these data questions?"
            if plural
            else f"What is the JSON formatted decomposition for this data question?"
        )

        prompt = pre.format(examples=examples, question=q, intro=intro)

        return prompt.strip()

    def make_var_prompt(self, q):

        prompt = Path(prmpt.__file__).parent.joinpath("decompose_vars.txt").read_text()

        indent = " " * 4

        qstr = indent + ("\n" + indent).join(str(e) for e in q)

        prompt = prompt.format(question=qstr)

        return prompt.strip()

    def decompose(self, question, prompt=None):
        from researchrobot.openai.completions import openai_one_completion

        if prompt is None:
            prompt = self.make_question_prompt(question)

        try:

            json_r = openai_one_completion(prompt, system=None, model=self.model_id)

        except Exception as e:
            print(f'ERROR for prompt\n"{prompt}"\n\n', e)
            raise
            return None

        # It still sometimes adds a preable to the JSON
        if self._is_multi(question):
            json_r = json_r[json_r.find("[") :]

            try:
                decomps = []
                for e, q in zip(json.loads(json_r), question):
                    dc = Decomposition.from_json(e)
                    dc.question = q
                    decomps.append(dc)

                return decomps
            except Exception as e:

                print("JSON Error", e)

                return json_r

        else:
            json_r = json_r[json_r.find("{") :]

            try:
                dc = Decomposition.from_json(json_r)
                dc.question = question
                return dc
            except Exception as e:
                print("JSON Error", e)

                return json_r

    def decompose_var(self, question, return_all=False):
        return self.decompose(question, prompt=self.make_var_prompt(question))
