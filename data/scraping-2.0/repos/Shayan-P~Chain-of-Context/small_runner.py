import json
import os
import pathlib
import random
import warnings
from concurrent.futures.thread import ThreadPoolExecutor

import torch
import openai
import datasets
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from eval.args import RunnerArguments, HFArguments, OAIArguments, GenerationArguments
from eval.evaluator import HFEvaluator, OAIEvaluator, _WARNING
from eval.tasks import ALL_TASKS, get_task

from dotenv import load_dotenv

load_dotenv()  # loads the api keys in .env

transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()

runner_args = RunnerArguments()
hf_args = HFArguments()
oai_args = OAIArguments()
gen_args = GenerationArguments()
args = HfArgumentParser([runner_args, hf_args, oai_args, gen_args]).parse_args()

task_name = "proofwriter-neurosymbolic-2shot"

args.task_name = task_name

args.max_length_generation = 3000
args.temperature = 0.8  # todo consider increasing the temperature?
args.openai_api_env_keys = ['OPENAI_API_KEY', 'OPENAI_API_KEY2', 'OPENAI_API_KEY3', 'OPENAI_API_KEY4', 'OPENAI_API_KEY5', 'OPENAI_API_KEY6']
args.model = 'gpt-4-0613' # 'gpt-3.5-turbo-16k-0613'  # 'gpt-3.5-turbo'  # hard coded model here
args.allow_code_execution = True

# todo remove debug suffix from here later
run_id = f"{args.model}_${task_name}_benchmark_coc"
args.save_generations_raw = True
args.save_generations_prc = True
args.save_references = True
args.save_results = True
args.save_context = True
args.save_dataset_indices = True

args.save_results_path = f'{run_id}_results.json'
args.save_context_path = f'{run_id}_context.json'
args.save_references_path = f'{run_id}_references.json'
args.save_generations_raw_path = f'{run_id}_generations_raw.json'
args.save_generations_prc_path = f'{run_id}_generations_prc.json'
args.save_dataset_indices_path = f'{run_id}_dataset_indices.json'
args.output_dir = 'outputs'

args.output_dir = pathlib.Path(os.getcwd()) / args.output_dir
args.save_generations_raw_path = args.output_dir / args.save_generations_raw_path
args.save_generations_prc_path = args.output_dir / args.save_generations_prc_path
args.save_references_path = args.output_dir / args.save_references_path
args.save_results_path = args.output_dir / args.save_results_path
args.save_context_path = args.output_dir / args.save_context_path
args.save_dataset_indices_path = args.output_dir / args.save_dataset_indices_path
args.save_generations_raw_path.parent.mkdir(parents=True, exist_ok=True)
args.save_generations_prc_path.parent.mkdir(parents=True, exist_ok=True)
args.save_references_path.parent.mkdir(parents=True, exist_ok=True)
args.save_results_path.parent.mkdir(parents=True, exist_ok=True)
args.save_dataset_indices_path.parent.mkdir(parents=True, exist_ok=True)

my_parent_task = get_task(task_name)


# this is a very bad engineering practice. don't do that at home!
class CustomSubTask(type(my_parent_task)):
    def __init__(self, *args, **kwargs):
        super(CustomSubTask, self).__init__(*args, **kwargs)
        self.stop_words.append("</OUTPUT>")

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        instructions = self.get_instructions()
        train = self.fewshot_examples()
        test = self.format_test_example(doc)
        prompt = "\n".join([instructions, train, test])
        return prompt

    def get_extra_context_prompt(self, generation):
        instruction_and_train = self.get_instruction_and_extra_context_examples()
        test = self.format_extra_context_example(generation)
        return "\n".join([instruction_and_train, test])

    def format_extra_context_example(self, generation):
        # todo maybe we should also include the doc itself (similar to the format_example?)
        return f"""
<INPUT>
<PREMISES>
{generation}
</PREMISES>
</INPUT>
<OUTPUT>
"""

    def get_instruction_and_extra_context_examples(self):
        return f"""
You will be given the premises for a first-order logic (FOL) problem.
The problem is to identify additional premises that are implicitly common sense from the ones given.
The premises are given in the form of a set of first-order logic sentences.
The task is to generate new common sense premises, text and FOL pairs, that would be common sense to someone reading the original premises.
These new common sense premises should reflect the nature of synonyms and antonyms, categorize proper names, and identify implicit characteristics from the ones provided.
Do not limit the amount of new premises generated in the output.
Expressions should be adhere to the format of the Python NLTK package logic module. Here are a couple examples:
<INPUT>
<PREMISES>
Premise: When a person reads a book, that person
gains knowledge.
FOL: all x. all y. (Person(x) & Reads(x, y) &
Book(y) -> Gains(x, Knowledge))
Premise: Harry read the book "Walden" by Henry
Thoreau.
FOL: Reads(Harry, Walden)
</PREMISES>
</INPUT>

<OUTPUT>
Premise: Harry is a human.
FOL: Person(Harry)
</OUTPUT>

<INPUT>
<PREMISES>
Premise: Heinrich Schmidt was a Nazi German
politician.
FOL: NaziGermanPolitician (HeinrichSchmidt)
</PREMISES>
</INPUT>

<OUTPUT>
Premise: Heinrich Schmidt was a Nazi
FOL: Nazi(HeinrichSchmidt)
Premise: Heinrich Schmidt was a German
FOL: German(HeinrichSchmidt)
Premise: Heinrich Schmidt was a Politician
FOL: Politician(HeinrichSchmidt)
</OUTPUT>

<INPUT>
<PREMISES>
Premise: Famine is bad
FOL: Bad(Famine)
</PREMISES>
</INPUT>
<OUTPUT>
Premise: Bad is not good.
FOL: Bad -> -Good
Premise: Cold is not warm
FOL: Cold -> -Warm
</OUTPUT>

After generating this output pick the top 5 premises that are the most relevant to the provided premises, making sure to prune any that contradict each other or the given input. Return these with the tags <CONTEXT> and </CONTEXT> in the same format as the output.
"""


class CustomOAIEvaluator(OAIEvaluator):
    def generate_text(self, task):
        dataset = task.get_dataset()
        n_tasks = self.args.limit if self.args.limit else len(dataset)
        indices = list(range(n_tasks))
        prompts = [task.get_prompt(dataset[i]) for i in indices]
        stops = [task.stop_words for _ in range(n_tasks)]

        with ThreadPoolExecutor() as executor:
            generations_raw = list(executor.map(self.get_completion, prompts, stops))

        # todo change n_sample of here to 1
        context_prompt = [task.get_extra_context_prompt(random.choice(generation)) for generation in generations_raw]

        with ThreadPoolExecutor() as executor:
            contexts = list(executor.map(self.get_completion, context_prompt, stops))

        # todo bad code. doesn't work for sample bigger than 1

        def extract_context_from_raw(text):
            # this is for the new context Ate added
            try:
                return text.split('<CONTEXT>')[1].split('</CONTEXT>')[0]
            except Exception as e:
                print("failed to get the <CONTEXT> part in", text)
                print('returning all of the generated thing')
                return text

        generations_prc = [
            [
                task.postprocess_generation(
                    extract_context_from_raw(contexts[i][j]) + '\n' + generations_raw[i][j], i, completion_only=True
                )
                for j in range(self.args.n_samples)
            ]
            for i in range(n_tasks)
        ]
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]
        return indices, generations_prc, generations_raw, contexts, references

    def generate_raw(self, task, doc):
        pass

    # uses task instead of task_name
    def evaluate(self, task):
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        indices, generations_prc, generations_raw, contexts, references = self.generate_text(task)
        if len(generations_prc[0]) != self.args.n_samples:
            generations_prc = [l[: self.args.n_samples] for l in generations_prc]
            warnings.warn(
                "Number of tasks wasn't proportional to number of devices, we removed extra predictions"
            )

        if not hasattr(self, "accelerator") or self.accelerator.is_main_process:
            if not self.args.generations_path:
                if self.args.save_generations_raw:
                    with open(self.args.save_generations_raw_path, "w") as fp:
                        json.dump(generations_raw, fp)
                        print("raw generations were saved")
                if self.args.save_generations_prc:
                    with open(self.args.save_generations_prc_path, "w") as fp:
                        json.dump(generations_prc, fp)
                        print("processed generations were saved")
                if self.args.save_context:
                    with open(self.args.save_context_path, "w") as fp:
                        json.dump(contexts, fp)
                        print("references were saved")
                if self.args.save_references:
                    with open(self.args.save_references_path, "w") as fp:
                        json.dump(references, fp)
                        print("references were saved")
                if self.args.save_dataset_indices:
                    with open(self.args.save_dataset_indices_path, "w") as fp:
                        json.dump({"task_name": self.args.task_name, "indices": indices}, fp)
                        print("dataset indices were saved")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            results = task.process_results(generations_prc, references)
            return results

    # uncomment this if you don't have access to LLM or you want to manually debug...
    def make_request(self, prompt, stop):
        print("please query this and tell me the answer:")
        print("stop words: ", stop)
        print(prompt)
        return super(CustomOAIEvaluator, self).make_request(prompt, stop)

        # print("paste here and type end at the end:")
        # res = ""
        # while True:
        #     s = input()
        #     if s.strip() == "end":
        #         break
        #     elif s.strip():
        #         res += s.strip() + "\n"
        # return [res] * self.args.n_samples  # copy instead of asking n times...


if __name__ == "__main__":
    task = CustomSubTask()

    args.n_samples = 1
    args.limit = 60  # hard coded so that we don't run all examples

    is_chat = True  # todo change this if you change model

    # todo uncomment this for checking the comparison
    # task = my_parent_task
    # evaluator = OAIEvaluator(args, chat=is_chat)
    # result = evaluator.evaluate(task_name)

    evaluator = CustomOAIEvaluator(args, chat=is_chat)
    result = evaluator.evaluate(task)

    print(result)
