import csv
import json
import ast
import os
import openai
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from rouge_score import rouge_scorer
from typing import List, Dict, Tuple
from datasets import logging, load_dataset, concatenate_datasets, Dataset
from config import JSON_SCHEMA

EXCLUDED_TASKS = ["cifar10_classification", "mnist_ascii"]
logging.set_verbosity_warning()
logging.disable_progress_bar()  # hide the progress bars everytime we load a dataset


class GenerationPipeline:
    def __init__(
        self,
        tasks_dir: str,
        train_dir: str,
        train_dir_gen: str,
        error_dir: str,
        task_seed_size: int,
        sk: str,
        filter_errors: str = "filter_errors.csv",
        error_file: str = "errors.txt",
        epochs: int = 40,
        max_generations: int = 200,
    ) -> None:
        """
        tasks_dir: name of file (csv) of tasks to augment
        train_dir: file path to directory to write data to
        train_dir_gen: file path to dump generated instances
        error_dir: file path to dump error related data
        task_seed_size: size of seed examples per task
        filter_errors: file name (csv) to write filter errors to
        sk: OpenAI api key
        """
        openai.api_key = sk
        self.train_dir = train_dir
        self.train_dir_gen = train_dir_gen
        self.error_dir = error_dir
        self.filter_errors = filter_errors
        self.error_file = error_file
        self.EPOCHS = epochs
        self.MAX_GENERATIONS = max_generations

        with open(tasks_dir, newline="") as f:
            reader = csv.reader(f)
            tasks = [row[-1] for row in reader]
        tasks = tasks[1:]
        f.close()
        tasks = list(filter(lambda task: task not in EXCLUDED_TASKS, tasks))

        # train test split wrt tasks
        idx = np.random.permutation(len(tasks))
        split = int(len(tasks) * 0.8)  # 80, 20 split
        train_idx, test_idx = idx[:split], idx[split:]
        self.train_tasks = [tasks[i] for i in train_idx]
        self.test_tasks = [tasks[i] for i in test_idx]

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        # # initialize seed task jsons in training directory
        self.missing_tasks = self._init_task_seeds(task_seed_size, train_dir)
        self.train_tasks = list(
            filter(lambda x: x not in self.missing_tasks, self.train_tasks)
        )

        # initialize error recording directory
        if not os.path.exists(self.error_dir):
            os.makedirs(self.error_dir)
            with open(f"{self.error_dir}/{self.filter_errors}", "a+") as f:
                f.write("task,parse_filters,field_check_filters,duplicate_filters")
            f.close()
            with open(f"{self.error_dir}/summary.csv", "a+") as f:
                f.write("iter,task,generations,generation_time")
            f.close()
            with open(f"{self.error_dir}/{self.error_file}", "a+") as f:
                f.write("Error Dumps")
            f.close()

        # initialize json files directory to dump generated examples into
        if not os.path.exists(self.train_dir_gen):
            os.makedirs(self.train_dir_gen)
        # initalize empty jsons to dump generated examples into
        for task in self.train_tasks:
            if not os.path.exists(f"{self.train_dir_gen}/{task}.json"):
                with open(f"{self.train_dir_gen}/{task}.json", "w") as f:
                    json.dump([], f)
                f.close()

        self.descr_df = pd.read_csv("./data/metadata/task_descriptions.csv")

    def _init_task_seeds(self, task_seed_size: int, train_dir: str) -> List[str]:
        """
        only augment training tasks

        train_dir: file path to directory to write data to
        task_seed_size: size of seed examples per task
        return missing tasks lists: [str]
        """
        missing_tasks = []
        for task in tqdm(self.train_tasks):
            path = f"{train_dir}/{task}.json"
            if os.path.exists(os.path.join(os.getcwd(), path)):  # already loaded
                continue
            try:
                D_task = load_dataset("tasksource/bigbench", task, split="train")
                k_idx = np.random.randint(
                    low=0, high=len(D_task), size=task_seed_size
                ).tolist()
                D_init = D_task.select(k_idx)
                is_generated = [False] * len(D_init)
                D_init = D_init.add_column("is_generated", is_generated)
                D_init = D_init.add_column("true_idx", k_idx)
                D_init.remove_columns("idx")
                D_init.to_json(path, orient="records", lines=False)
            except:
                missing_tasks.append(task)
        return missing_tasks

    def augment(
        self,
        k: int,
        total_qs: int,
        SKIP_TASKS: List[str] = [],
        TARG_TASKS: List[str] = [],
    ) -> None:
        """
        for all train tasks:
        (0) Sample k examples from task
          * format k examples into prompt
        (1) GPT to generate additional examples
          * process output into json (inputs, m)
          * deterministic generation mode (temp=0, top_p=1)
        (2) Filter additional examples
        (3) add augmented examples into tasks super set

        k: amount of in-context examples for generation
        total_qs: total number of question to generate (including in-context
        SKIP_TASKS: if defined, skip these tasks
        TARG_TASKS: if defined, only augment these tasks
        examples)
        """
        train_tasks = os.listdir(self.train_dir)
        train_tasks = [t[:-5] for t in train_tasks]  # get rid of .json
        for _ in range(self.EPOCHS):
            for i, task in enumerate(tqdm(train_tasks)):
                start = time.time()
                if SKIP_TASKS and task in SKIP_TASKS:
                    print(f"skipping task: {task}")
                    continue
                if TARG_TASKS and task not in TARG_TASKS:
                    print(f"skipping task (not in target): {task}")
                    continue

                print(f"Generating examples for {task}")
                D_task = load_dataset(
                    "json", data_files=f"{self.train_dir}/{task}.json", split="train"
                )
                D_task_aug = load_dataset(
                    "json",
                    data_files=f"{self.train_dir_gen}/{task}.json",
                    split="train",
                )
                if len(D_task_aug) > self.MAX_GENERATIONS:
                    print(f"Reached augmented capacity: {task}")
                    continue

                # sample indeces for selection, treat len(D_task) as offset
                k_idx = np.random.randint(
                    low=0, high=len(D_task) + len(D_task_aug), size=k
                ).tolist()
                k_idx_real = list(filter(lambda x: x < len(D_task), k_idx))
                k_idx_aug = list(
                    np.array(list(filter(lambda x: x >= len(D_task), k_idx)))
                    - len(D_task)
                )
                # select examples from both real and augmented task dataset
                D_subset = D_task.select(k_idx_real)
                D_subset = D_subset.remove_columns(
                    ["multiple_choice_scores", "idx", "is_generated", "true_idx"]
                )
                D_subset_aug = D_task_aug.select(k_idx_aug)
                if len(D_subset_aug) > 0:
                    D_subset_aug = D_subset_aug.cast(D_subset.features)
                D_examples = concatenate_datasets([D_subset, D_subset_aug], axis=0)
                D_examples = D_examples.shuffle()
                # construct prompt
                prompt = self.to_prompt(D_examples, total_qs, task)
                # make request
                msg_content = self.make_request(prompt)
                # process request
                filter_summary = self.process_request(msg_content, task)
                # record error summary
                end = time.time()
                self._record_stats(i, task, filter_summary, round(end - start, 3))

    def _record_stats(
        self, i: int, task: str, filter_summary: Dict, gen_time: int
    ) -> None:
        total_errors = sum(filter_summary.values())
        with open(f"{self.error_dir}/{self.filter_errors}", "a+") as f:
            # task, parse filter, rl filter, fc filter <-- columns

            filter_summary["task"] = task
            dict_writer = csv.DictWriter(
                f,
                fieldnames=[
                    "task",
                    "parse_filters",
                    "field_check_filters",
                    "duplicate_filters",
                ],
            )
            dict_writer.writerow(filter_summary)
        f.close()
        with open(f"{self.error_dir}/summary.csv", "a+") as f:
            iter_summary = {
                "iter": i,
                "task": task,
                "generations": 6 - total_errors,  # 10 <- total_qs
                "generation_time": gen_time,
            }
            dict_writer = csv.DictWriter(
                f,
                fieldnames=["iter", "task", "generations", "generation_time"],
            )
            dict_writer.writerow(iter_summary)
        f.close()

    def to_prompt(self, dataset: Dataset, total_qs: int, task: str) -> str:
        """
        dataset: subset to embed as context for prompt
        total_qs: total questions to generate including embedded questions
        task: task name
        """
        keys = ["inputs", "targets", "multiple_choice_targets"]
        # each key has same amount of values want to map each value of each key with each other
        ds_dict = dataset.to_dict()
        n_examples = len(ds_dict["inputs"])

        examples = []
        for i in range(n_examples):
            example = {}
            for key in keys:
                example[key] = ds_dict[key][i]
            examples.append(example)
        task_descr = self.descr_df[(self.descr_df["task"] == task)][
            "description"
        ].values[0]
        prompt = f"DESCRIPTION: {task_descr}\n"
        prompt += f"SCHEMA:\n{JSON_SCHEMA}\n"
        prompt += f"Generate a series of {total_qs} diverse questions related to DESCRIPTION in a valid JSON format that follows SCHEMA."
        for i, e in enumerate(examples):
            prompt += f"\nQuestion {i}:\n{e}"
        prompt += f"\nQuestion {n_examples}:"
        return prompt

    def make_request(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """
        Assumes only sending one prompt (e.g., not sending list of prompts)

        prompt: includes in-context examples
        model: model used for generation
        """
        msg_content = ""
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                presence_penalty=0.8,
            )
            msg_content = completion["choices"][0]["message"]["content"]
        except openai.error.OpenAIError as e:
            print(f"OpenAIError: {e}.")

        return msg_content

    def process_request(self, msg_content: str, task: str) -> Dict:
        """
        processes msg_content from GPT
        (1) parse msg_content to python dict
        (2) applying filters to dictionary examples
        (4) write filtered messages to appropriate task db (json)

        res: message content from GPT
        task: name of task generated questions correspond to
        """
        questions_parsed, parsed_errs = self._generation_to_dicts(msg_content)
        questions_fc_pass, fc_errs = self._field_check(questions_parsed)
        added_questions, dup_errs = self._duplicate_check(task, questions_fc_pass)

        filter_summary = {
            "parse_filters": parsed_errs,
            "field_check_filters": fc_errs,
            "duplicate_filters": dup_errs,
        }

        with open(f"{self.train_dir_gen}/{task}.json", "r") as f:
            task_ds = json.load(f)
        f.close()

        task_ds.extend(added_questions)

        with open(f"{self.train_dir_gen}/{task}.json", "w") as f:
            json.dump(task_ds, f)
        f.close()

        return filter_summary

    def _str_to_dict(self, q: str) -> Dict:
        """escapes single/double quotes in the inputs field of the string and converts it to a dictionary"""

        def replacer(s: re.Match) -> str:
            s = re.sub(r"(?<!\\)\'", "\\'", s.group(0))
            s = re.sub(r"(?<!\\)\"", '\\"', s)
            return s

        # define the regular expression pattern to match single and double quotes
        pattern = r"""(?<=["']inputs["']: ["']).*(?=["'], ["']targets)"""

        # replace the matched quotes with escaped quotes
        escaped_str = re.sub(pattern, replacer, q)

        return ast.literal_eval(escaped_str)

    def _generation_to_dicts(self, msg_content: str) -> Tuple[List[Dict], int]:
        """
        returns list of dictionary representing generated questions
        """

        # change split word? what if 'Question: \d' in examples
        generate_qs = re.split("Question \d:", msg_content)
        generated_dict_examples, errs = [], 0
        for i, q in enumerate(generate_qs):
            if i == 0:  # end of prompt = "... Question i:"
                try:
                    msg_dict = self._str_to_dict(q)
                    generated_dict_examples.append(msg_dict)
                except:
                    self.dump_error(q, self.error_file)
                    errs += 1
            else:  # cut off Question:\n that comes before the {<json_question>}
                q = q[q.find("\n") :]
                q = q.rstrip("\n")
                try:
                    msg_dict = self._str_to_dict(q)
                    generated_dict_examples.append(msg_dict)
                except:
                    self.dump_error(q, self.error_file)
                    errs += 1

        return generated_dict_examples, errs

    def _field_check(self, questions: List[Dict]) -> Tuple[List[Dict], int]:
        """
        questions: generated question list in dict format
        """
        keys = set(["inputs", "targets", "multiple_choice_targets"])
        out, errs = [], 0
        for q in questions:
            if (
                keys == set(q.keys())
                and isinstance(q["inputs"], str)
                and isinstance(q["targets"], list)
                and isinstance(q["multiple_choice_targets"], list)
                and all(isinstance(val, str) for val in q["targets"])
                and all(isinstance(val, str) for val in q["multiple_choice_targets"])
            ):
                out.append(q)
            else:
                self.dump_error(q, self.error_file)
                errs += 1

        return out, errs

    def _duplicate_check(
        self, task: str, questions: List[Dict]
    ) -> Tuple[List[Dict], int]:
        """
        Filter out exact duplicates based on inputs field
        """
        # get current examples in db
        with open(f"{self.train_dir}/{task}.json", "r") as f:
            ds_init = json.load(f)
        f.close()
        with open(f"{self.train_dir_gen}/{task}.json", "r") as f:
            existing_generated_questions = json.load(f)
        f.close()

        ds_init.extend(existing_generated_questions)

        ds_inputs = set([question["inputs"] for question in ds_init])
        generated_inputs = [question["inputs"] for question in questions]

        excluded_qs, errs = [], 0
        for i, generated_input in enumerate(generated_inputs):
            if generated_input in ds_inputs:
                excluded_qs.append(i)
                self.dump_error(generated_input, self.error_file)
                errs += 1

        return [q for i, q in enumerate(questions) if i not in excluded_qs], errs

    def _rougel_check(self, task: str, questions: List[Dict]) -> Tuple[List[Dict], int]:
        """
        duplicate check, assumes questions pass _field_check
        task: task name
        questions: generated question list in dict format (e.g.,
        {inputs:<str>, targets:<str>, multiple_choice_targets:<str>})
        """
        # get current examples in db
        with open(f"{self.train_dir}/{task}.json", "r") as f:
            ds_init = json.load(f)
        f.close()
        with open(f"{self.train_dir_gen}/{task}.json", "r") as f:
            existing_generated_questions = json.load(f)
        f.close()

        ds_init.extend(existing_generated_questions)

        ds_inputs = [question["inputs"] for question in ds_init]
        generated_inputs = [question["inputs"] for question in questions]

        scorer = rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=False
        )  # same as self-instruct setting
        scores, excluded_qs, errs = defaultdict(list), set(), 0
        for i, existing_q in enumerate(ds_inputs):
            for j, new_q in enumerate(generated_inputs):
                score = scorer.score(new_q, existing_q)["rougeL"].fmeasure
                scores[j].append((i, score))
                if score > 0.99:
                    self.dump_error(existing_q, self.error_file)
                    excluded_qs.add(j)
                    errs += 1

        return [q for i, q in enumerate(questions) if i not in excluded_qs], errs

    def dump_error(self, res: str, error_file: str):
        with open(f"{self.error_dir}/{error_file}", "a+") as f:
            f.write(f"{res}\n")
        f.close()
