from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import random
import requests
import torch.nn as nn
import torch
import numpy as np
from filelock import FileLock
import os
import pickle
from collections import defaultdict
from eval_utils import process_skill_strings
from threading import Lock
from utils import AttrDict


class LargeLanguageModel:
    starter = "Task Steps:\n"
    summary_start = "Instructions: give a high-level description for the following steps describing common household tasks.\n\n"
    summary_prompt_start = summary_start
    summary_prompt_start += starter
    summary_prompt_start += "1. Pick up the keys on the center table.\n"
    summary_prompt_start += "2. Put the keys in the box.\n"
    summary_prompt_start += "3. Pick up the box with keys.\n"
    summary_prompt_start += (
        "4. Put the box with keys on the sofa close to the newspaper.\n"
    )
    summary_prompt_start += (
        "Summary: Put the box with keys on the sofa next to the newspaper.\n"
    )

    summary_prompt_start += starter
    summary_prompt_start += "1. Pick up the knife from in front of the tomato.\n"
    summary_prompt_start += "2. Cut the lettuce on the counter.\n"
    summary_prompt_start += (
        "3. Set the knife down on the counter in front of the toaster.\n"
    )
    summary_prompt_start += "4. Pick up a slice of the lettuce from the counter.\n"
    summary_prompt_start += "5. Put the lettuce slice in the refrigerator. take the lettuce slice out of the refrigerator.\n"
    summary_prompt_start += (
        "6. Set the lettuce slice on the counter in front of the toaster.\n"
    )
    summary_prompt_start += (
        "Summary: Cool a slice of lettuce and put it on the counter.\n"
    )

    summary_prompt_start += starter
    summary_prompt_start += "1. Pick up the book on the table, in front of the chair.\n"
    summary_prompt_start += "2. Place the book on the left cushion of the couch.\n"
    summary_prompt_start += "Summary: Put the book on the table on the couch.\n"

    summary_prompt_start += starter
    summary_prompt_start += "1. Pick up the fork from the table.\n"
    summary_prompt_start += "2. Put the fork in the sink and fill the sink with water, then empty the water from the sink and remove the fork.\n"
    summary_prompt_start += "3. Put the fork in the drawer.\n"
    summary_prompt_start += (
        "Summary: Rinse the fork in the sink and then put it in a drawer.\n"
    )

    summary_prompt_start += starter
    summary_prompt_start += "1. Take the box of tissues from the makeup vanity.\n"
    summary_prompt_start += "2. Put the tissues on the barred rack.\n"
    summary_prompt_start += "3. Take the box of tissues from the top of the toilet.\n"
    summary_prompt_start += "4. Put the tissues on the barred rack.\n"
    summary_prompt_start += "Summary: Put two boxes of tissues on the barred rack.\n"

    summary_prompt_start += starter
    summary_prompt_start += "1. Pick up the glass from the sink.\n"
    summary_prompt_start += "2. Heat the glass in the microwave.\n"
    summary_prompt_start += "3. Put the glass on the wooden rack.\n"
    summary_prompt_start += (
        "Summary: Put a heated glass from the sink onto the wooden rack.\n"
    )

    summary_prompt_start += starter
    summary_prompt_start += "1. Pick up the box from the far side of the bed.\n"
    summary_prompt_start += "2. Hold the box and turn on the lamp.\n"
    summary_prompt_start += (
        "Summary: Look at the box from the far side of the bed under the lamp light.\n"
    )

    # start of last 7
    #summary_prompt_start += starter
    #summary_prompt_start += "1. Pick up the pencil on the desk, nearest to the clock.\n"
    #summary_prompt_start += "2. Put the pencil in the mug on the desk.\n"
    #summary_prompt_start += "3. Pick up the mug from the desk.\n"
    #summary_prompt_start += "4. Place the mug on the desk, in front of the computer.\n"
    #summary_prompt_start += "Summary: Put the pencil on the desk in the mug and place it on the desk in front of the computer.\n"

    #summary_prompt_start += starter
    #summary_prompt_start += "1. Pick up the apple on the counter.\n"
    #summary_prompt_start += "2. Place the apple in the fridge and close the door. wait a moment and then take the apple out.\n"
    #summary_prompt_start += "3. Place the apple in the microwave.\n"
    #summary_prompt_start += "Summary: Place a chilled apple in a microwave.\n"

    #summary_prompt_start += starter
    #summary_prompt_start += "1. Take the towel from the towel ring on the wall.\n"
    #summary_prompt_start += "2. Put the towel down in the sink.\n"
    #summary_prompt_start += (
    #    "Summary: Put the towel from the towel ring into the sink.\n"
    #)

    #summary_prompt_start += starter
    #summary_prompt_start += "1. Open the fridge, pick up the green cup next to the tomato and close the fridge.\n"
    #summary_prompt_start += "2. Put the cup in the sink, turn on the water for a few seconds, turn it off and pick up the cup again.\n"
    #summary_prompt_start += "3. Put the cup on the left edge of the cabinet.\n"
    #summary_prompt_start += (
    #    "Summary: Put a washed cup from the fridge onto the left edge of the cabinet.\n"
    #)

    #summary_prompt_start += starter
    #summary_prompt_start += "1. Pick up the keys that are closest to the edge, in the center of the couch.\n"
    #summary_prompt_start += (
    #    "2. Place the keys on the corner of the table, closest to the fireplace.\n"
    #)
    #summary_prompt_start += "3. Pick up the keys close to the edge of the shelf.\n"
    #summary_prompt_start += (
    #    "4. Place the keys on the table, between the other set of keys and the watch.\n"
    #)
    #summary_prompt_start += (
    #    "Summary: Put two sets of keys, from the couch and table, on the table.\n"
    #)

    #summary_prompt_start += starter
    #summary_prompt_start += "1. Pick up the knife off of the table.\n"
    #summary_prompt_start += "2. Slice the bread on the table.\n"
    #summary_prompt_start += "3. Open the cupboard and place the knife inside.\n"
    #summary_prompt_start += "4. Pick up the slice of bread.\n"
    #summary_prompt_start += "5. Open the microwave and heat the slice of bread.\n"
    #summary_prompt_start += "6. Place the toast inside the bin.\n"
    #summary_prompt_start += (
    #    "Summary: Warm a slice of bread in the microwave and place it in the bin.\n"
    #)

    #summary_prompt_start += starter
    #summary_prompt_start += (
    #    "1. Pick up the clock towards the back on top of the desk.\n"
    #)
    #summary_prompt_start += "2. Turn on the lamp.\n"
    #summary_prompt_start += (
    #    "Summary: Look at the clock, from on top of the desk, under the light.\n"
    #)
    summary_prompt_start += starter
    summary_prompt_mid = lambda self, index, text: f"{index+1}. {text}\n"
    summary_prompt_end = "Summary:"

    all_next_skill_prompt_start = (
        "Examples of common household tasks and their descriptions: \n\n"
    )
    all_next_skill_prompt_start = summary_prompt_start.replace(
        summary_start, all_next_skill_prompt_start
    )
    # replace summary with task
    all_next_skill_prompt_start = all_next_skill_prompt_start.replace(
        "Summary: ", "Task: "
    )
    # remove the last starter
    all_next_skill_prompt_start = all_next_skill_prompt_start[: -len(starter)]
    # replace with new text
    all_next_skill_prompt_start += "\nPredict the next skill correctly by choosing from the following next skills: "

    all_next_skill_aggregate_skills = (
        lambda self, text: f"{text.replace(text[-1], ';')}"
    )

    USE_LOGPROB_PROMPT_FOR_LOGPROB = False

    def __init__(self, config):
        assert (
            "opt" in config.llm_model
            or "gpt" in config.llm_model
            or "alpaca" in config.llm_model
            or "llama" in config.llm_model
            or "None" in config.llm_model
        ), "No tokenizer support for non-gpt/opt models"
        super().__init__()
        self.config = config
        self.llm_gpus = config.llm_gpus
        # self.nli_gpu = config.nli_gpu
        self.llm_max_new_tokens = config.llm_max_new_tokens
        self.llm_batch_size = config.llm_batch_size
        self.generate_next_skill_with_codex = config.generate_next_skill_codex
        # self.nli_classifier = pipeline(
        #    "zero-shot-classification",
        #    model="facebook/bart-large-mnli",
        #    device=self.nli_gpu,
        # )
        if config.llm_model != "None":
            tokenizer_cls = AutoTokenizer
            model_cls = AutoModelForCausalLM
            if "alpaca" in config.llm_model or "llama" in config.llm_model:
                tokenizer_cls = LlamaTokenizer
                model_cls = LlamaForCausalLM
                model_size = "7B" if "7b" in config.llm_model.lower() else "13B"
                path = '/data/jesse/hf_llama_weights/' + model_size
                #path = '/data2/jesse/hf_llama_weights/' + model_size
                config.llm_model = path

            self.tokenizer = tokenizer_cls.from_pretrained(
                config.llm_model,
                model_max_length=2048,
                # use fast is false to avoid threading issue
                use_fast=False,
            )

            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
            #self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model = model_cls.from_pretrained(
                config.llm_model,
                pad_token_id=self.tokenizer.pad_token_id,
                device_map="auto",
                load_in_8bit=True,
                torch_dtype=torch.float16,
            )
            #if torch.__version__ >= "2":
            #    self.model = torch.compile(self.model)
            # if len(self.llm_gpus) == 1:
            #    self.model = AutoModelForCausalLM.from_pretrained(
            #        config.llm_model,
            #        torch_dtype=torch.float16,
            #        pad_token_id=self.tokenizer.eos_token_id,
            #        # device_map="balanced_low_0",
            #        # device_map="auto",
            #    ).to(self.llm_gpus[0])
            # else:
            #    self.model = AutoModelForCausalLM.from_pretrained(
            #        config.llm_model,
            #        torch_dtype=torch.float16,
            #        pad_token_id=self.tokenizer.eos_token_id,
            #        device_map="auto",
            #    )
        # memory_dict = {GPU: "0GiB" for GPU in config.gpus}
        # device_map = infer_auto_device_map(my_model, max_memory={0: "10GiB", 1: "10GiB", "cpu": "30GiB"})
        # if len(self.llm_gpus) > 1:
        #    # split device map across available GPUs to leave enough room for inference with larger batch sizes
        #    max_memory_dict = {
        #        gpu: 1e9 for gpu in self.llm_gpus
        #    }  # 1e9 = 1GB so device map isn't just put on one gpu
        #    auto_device_map = infer_auto_device_map(
        #        self.model, dtype=torch.float16, max_memory=max_memory_dict
        #    )
        #    balanced_device_map = self.evenly_split_device_map(
        #        auto_device_map, self.llm_gpus
        #    )
        #    dispatch_model(self.model, device_map=balanced_device_map)
        # else:
        #    self.model.to(self.llm_gpus[0])
        self.lock = Lock()
        self.next_skill_top_p = 0.8
        #self.next_skill_temp = 0.5
        self.next_skill_temp = self.config.llm_next_skill_temp
        self.summary_temp = self.config.llm_summary_temp
        #if config.llm_model == "None":
        #    config.llm_model = "GPT"
        #self.summary_cache = defaultdict(list)
        #self.next_skill_cache = defaultdict(list)
        #self.next_skill_cache_location = f"next_skill_cache_{config.llm_model}.pkl"
        #self.summary_cache_location = f"summary_cache_GPT.pkl"

    def update_caches(self):
        with FileLock(self.next_skill_cache_location + ".lock"):
            if os.path.exists(self.next_skill_cache_location):
                with open(self.next_skill_cache_location, "rb") as f:
                    cache = pickle.load(f)
                    self.next_skill_cache.update(cache)
        with FileLock(self.summary_cache_location + ".lock"):
            if os.path.exists(self.summary_cache_location):
                with open(self.summary_cache_location, "rb") as f:
                    cache = pickle.load(f)
                    self.summary_cache.update(cache)

    def extract_info_from_caches(self, prompts):
        unextracted_prompts = []
        extracted_rets = []
        for prompt in prompts:
            if prompt in self.next_skill_cache and len(self.next_skill_cache[prompt]) >= 5:
                extracted_rets.append(self.next_skill_cache[prompt])


    def evenly_split_device_map(
        self, device_map, available_gpus, layers_limit=float("inf")
    ):
        # set(['.'.join(item[0].split('.')[:4]) for item in self.model.named_modules()])
        balanced_device_map = {}
        total_items = min(len(device_map), layers_limit)
        # available_gpus = [i for i in range(torch.cuda.device_count())]
        items_per_gpu = int(total_items / len(available_gpus))
        gpus_to_use = []
        for gpu in available_gpus:
            gpus_to_use.extend([gpu] * items_per_gpu)
        remaining_items = total_items - len(available_gpus) * items_per_gpu
        curr_gpu = 0
        for i in range(remaining_items):
            gpus_to_use.append(curr_gpu)
            curr_gpu = (curr_gpu + 1) % len(available_gpus)
        if layers_limit < len(device_map):
            cpus = (len(device_map) - layers_limit) * ["cpu"]
            gpus_to_use.extend(cpus)
        for i, name in enumerate(device_map.keys()):
            balanced_device_map[name] = gpus_to_use[i]
        return balanced_device_map

    def _get_logprobs_hf(
            self,
            output_dict,
    ):
        scores_float32 = [score.float() for score in output_dict.scores]
        scores = self.model.compute_transition_scores(output_dict.sequences, scores_float32, normalize_logits=True)
        return torch.mean(scores, dim=-1).cpu()
    
    def _get_non_generated_logprobs_hf(
        self,
        input_prompt_input_ids: torch.Tensor,
        input_prompt_attn_mask: torch.Tensor,
        second_skill_attn_mask: torch.Tensor,
    ):
        second_skill_start_pos = second_skill_attn_mask.sum(-1)
        with torch.no_grad():
            with self.lock:
                logits = (
                    self.model(
                        input_prompt_input_ids.to(self.llm_gpus[0]),
                        attention_mask=input_prompt_attn_mask.to(self.llm_gpus[0]),
                        return_dict=True,
                    )
                    .logits.cpu()
                    .float()
                )
        input_ids = input_prompt_input_ids
        if self.tokenizer.bos_token_id is not None:
            # the start token is attended to
            second_skill_start_pos -= 1
            # every logit but the last one because the logits correspond to distributions over the NEXT token given the token at the position
            logits = logits[:, :-1]
            # shifted_input_ids to disregard start token
            input_ids = input_prompt_input_ids[:, 1:]
        logprobs = torch.log_softmax(logits, dim=-1)
        token_specific_logprobs = logprobs.gather(2, input_ids.unsqueeze(2)).squeeze(2)
        token_logprobs = []
        for i in range(len(second_skill_start_pos)):
            token_logprobs.append(
                torch.mean(token_specific_logprobs[i, -second_skill_start_pos[i] :])
                # torch.sum(token_specific_logprobs[i, -second_skill_start_pos[i] :])
            )
        return torch.tensor(token_logprobs)


    def preprocess_llm_inputs_for_summarization(self, all_annotations: list[list]):
        modified_prompts = []
        for primitive_annotations in all_annotations:
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(primitive_annotations)
            ]
            modified_prompts.append(
                self.summary_prompt_start
                + "".join(with_mid_annotations)
                + self.summary_prompt_end
            )
        all_tokenized_prompts = self.tokenizer(
            modified_prompts, padding=True, truncation=True, return_tensors="pt"
        )
        return all_tokenized_prompts

    def preprocess_llm_inputs_for_logprob_summary_prompt(
        self, first_annotations: list[list[str]], second_annotations: list[list[str]],
    ):
        modified_prompts_without_end = []
        only_part_twos = []
        for prompt_part_one, prompt_part_two in zip(
            first_annotations, second_annotations
        ):
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(prompt_part_one)
            ]
            next_i = len(prompt_part_one)
            # second_with_mid = self.summary_prompt_mid(next_i, prompt_part_two)

            second_with_mid_annotations = [
                self.summary_prompt_mid(next_i + i, annotation)
                for i, annotation in enumerate(prompt_part_two)
            ]
            modified_prompts_without_end.append(
                self.summary_prompt_start
                + "".join(with_mid_annotations)
                + "".join(second_with_mid_annotations)
                # + self.summary_prompt_end
            )
            only_part_two = (
                " "
                + prompt_part_two[0]
                + "\n"
                + "".join(second_with_mid_annotations[1:])
            )
            only_part_twos.append(only_part_two)
            # only_part_twos.append(" " + prompt_part_two + "\n")
            # only_part_twos.append(second_with_mid_annotations + "\n")
        all_tokenized_prompts_no_end = self.tokenizer(
            modified_prompts_without_end,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized_second_parts = self.tokenizer(
            only_part_twos, padding=True, truncation=True, return_tensors="pt"
        )
        # print(modified_prompts_without_end[0])
        return (
            all_tokenized_prompts_no_end,
            tokenized_second_parts,
        )

    def preprocess_llm_inputs_for_logprob(
        self, first_annotations: list[str], second_annotations: list[str],
    ):
        # modified_prompts = []
        modified_prompts_without_end = []
        only_part_twos = []
        for prompt_part_one, prompt_part_two in zip(
            first_annotations, second_annotations
        ):
            prompt_part_one = prompt_part_one.lower()
            prompt_part_two = prompt_part_two.lower()
            # modified_prompts.append(
            #    self.logprob_prompt_start
            #    + prompt_part_one
            #    + self.logprob_prompt_mid
            #    + prompt_part_two
            #    + self.logprob_prompt_end
            # )
            modified_prompts_without_end.append(
                self.logprob_prompt_start
                + prompt_part_one
                + self.logprob_prompt_mid
                + prompt_part_two
            )
            if len(self.logprob_prompt_mid) > 0 and self.logprob_prompt_mid[-1] == " ":
                only_part_twos.append(" " + prompt_part_two)
            else:
                only_part_twos.append(prompt_part_two)

        all_tokenized_prompts_no_end = None
        tokenized_second_parts = None
        all_tokenized_prompts_no_end = self.tokenizer(
            modified_prompts_without_end,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized_second_parts = self.tokenizer(
            only_part_twos, padding=True, truncation=True, return_tensors="pt"
        )
        # print(modified_prompts_without_end[0])
        return (
            all_tokenized_prompts_no_end,
            tokenized_second_parts,
        )

    def query_logprobs(self, first_annotation, sample_second_annotations):
        all_logprobs = []
        all_first_annotations = [first_annotation] * len(sample_second_annotations)
        for i in range(0, len(sample_second_annotations), self.llm_batch_size):
            (
                tokenized_prompt_no_end,
                tokenized_second_part,
            ) = self.preprocess_llm_inputs_for_logprob_summary_prompt(
                all_first_annotations[i: i + self.llm_batch_size], sample_second_annotations[i: i + self.llm_batch_size]
            )
            batch_prompt_annotation_ids = tokenized_prompt_no_end.input_ids
            batch_prompt_annotation_attn_mask = tokenized_prompt_no_end.attention_mask
            batch_second_annotation_attn_mask = tokenized_second_part.attention_mask
            batch_logprobs = self._get_non_generated_logprobs_hf(
                batch_prompt_annotation_ids,
                # batch_second_annotation_ids,
                batch_prompt_annotation_attn_mask,
                batch_second_annotation_attn_mask,
            )
            all_logprobs.append(batch_logprobs)
        if len(all_logprobs) > 1:
            all_logprobs = torch.cat(all_logprobs, dim=0)
        else:
            all_logprobs = all_logprobs[0]
        return all_logprobs.cpu()

    def process_hf_generation(self, choice):
        if isinstance(choice["sequences"], torch.Tensor):
            generated_tokens = choice["sequences"][:, -self.llm_max_new_tokens :].cpu()
        else:
            # jax support
            generated_tokens = np.asarray(
                choice["sequences"][:, -self.llm_max_new_tokens :]
            )
        model_texts = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        generated_texts = []
        special_eos = "."
        for i in range(len(model_texts)):
            model_text = model_texts[i]
            if special_eos in model_text:
                model_text = model_text[: model_text.index(special_eos)]
            # reject bad responses
            if len(model_text) <= 5:
                return False
            generated_texts.append(model_text.strip())
        return process_skill_strings(generated_texts)

    def preprocess_llm_inputs_for_logprob_generation(
        self, first_annotations: list[list[str]]
    ):
        modified_prompts_without_end = []
        for prompt_part_one in first_annotations:
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(prompt_part_one)
            ]
            next_i = len(prompt_part_one)
            modified_prompts_without_end.append(
                self.summary_prompt_start
                + "".join(with_mid_annotations)
                + str(next_i + 1)
                + "."
            )
        all_tokenized_prompts_no_end = self.tokenizer(
            modified_prompts_without_end,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # print(len(modified_prompts_without_end))
        # print(modified_prompts_without_end[0])
        return all_tokenized_prompts_no_end

    def preprocess_llm_inputs_for_choosing_next_skill_generation(
        self, first_annotations: list[list[str]], all_possible_skills: list[list[str]]
    ):
        modified_prompts_without_end = []
        for prompt_part_one, available_next_skills in zip(
            first_annotations, all_possible_skills
        ):
            # randomize the order of available next skills
            random.shuffle(available_next_skills.copy())
            all_next_skill_sentence = (
                " ".join(
                    self.all_next_skill_aggregate_skills(skill)
                    for skill in process_skill_strings(available_next_skills)
                )
                + "\n"
            )
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(prompt_part_one)
            ]
            next_i = len(prompt_part_one)
            modified_prompts_without_end.append(
                self.all_next_skill_prompt_start
                + all_next_skill_sentence
                + self.starter
                + "".join(with_mid_annotations)
                + str(next_i + 1)
                + "."
            )
            # print(modified_prompts_without_end)
        all_tokenized_prompts_no_end = self.tokenizer(
            modified_prompts_without_end,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return all_tokenized_prompts_no_end

    def preprocess_llm_inputs_for_logprob_summary_prompt_all_skills(
        self,
        first_annotations: list[list[str]],
        all_possible_skills: list[list[str]],
        second_annotations: list[list[str]],
    ):
        modified_prompts_without_end = []
        only_part_twos = []
        for (prompt_part_one, prompt_part_two, available_next_skills,) in zip(
            first_annotations, second_annotations, all_possible_skills
        ):
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(prompt_part_one)
            ]
            next_i = len(prompt_part_one)
            # randomize the order of available next skills
            random.shuffle(available_next_skills.copy())
            all_next_skill_sentence = (
                " ".join(
                    self.all_next_skill_aggregate_skills(skill)
                    for skill in process_skill_strings(available_next_skills)
                )
                + "\n"
            )

            second_with_mid_annotations = [
                self.summary_prompt_mid(next_i + i, annotation)
                for i, annotation in enumerate(prompt_part_two)
            ]
            modified_prompts_without_end.append(
                self.all_next_skill_prompt_start
                + all_next_skill_sentence
                + self.starter
                + "".join(with_mid_annotations)
                + "".join(second_with_mid_annotations)
            )
            only_part_two = (
                " "
                + prompt_part_two[0]
                + "\n"
                + "".join(second_with_mid_annotations[1:])
            )
            only_part_twos.append(only_part_two)
        all_tokenized_prompts_no_end = self.tokenizer(
            modified_prompts_without_end,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized_second_parts = self.tokenizer(
            only_part_twos, padding=True, truncation=True, return_tensors="pt"
        )
        return (
            all_tokenized_prompts_no_end,
            tokenized_second_parts,
        )

    def query_logprobs_with_other_skills(
        self,
        first_annotation: list[str],
        sample_second_annotations: list[list[str]],
        all_possible_skills: list[str],
    ):
        all_first_annotations = [first_annotation] * len(sample_second_annotations)
        all_possible_skills_repeated = [all_possible_skills] * len(
            sample_second_annotations
        )
        all_logprobs = []
        for i in range(0, len(sample_second_annotations), self.llm_batch_size):
            (
                tokenized_prompt_no_end,
                tokenized_second_part,
            ) = self.preprocess_llm_inputs_for_logprob_summary_prompt_all_skills(
                all_first_annotations[i: i+ self.llm_batch_size],
                all_possible_skills_repeated,
                sample_second_annotations[i: i+ self.llm_batch_size],
            )
            batch_prompt_annotation_ids = tokenized_prompt_no_end.input_ids
            batch_prompt_annotation_attn_mask = tokenized_prompt_no_end.attention_mask
            batch_second_annotation_attn_mask = tokenized_second_part.attention_mask
            batch_logprobs = self._get_non_generated_logprobs_hf(
                batch_prompt_annotation_ids,
                batch_prompt_annotation_attn_mask,
                batch_second_annotation_attn_mask,
            )
            all_logprobs.append(batch_logprobs)
        if len(all_logprobs) > 1:
            all_logprobs = torch.cat(all_logprobs, dim=0)
        else:
            all_logprobs = all_logprobs[0]
        return all_logprobs.cpu()

    def generate_next_skill_with_other_skills_codex(
        self,
        first_annotations: list[list[str]],
        all_possible_skills: list[list[str]],
        num_generations,
    ):
        modified_prompts_without_end = []
        for prompt_part_one, available_next_skills in zip(
            first_annotations, all_possible_skills
        ):
            # randomize the order of available next skills
            random.shuffle(available_next_skills.copy())
            all_next_skill_sentence = (
                " ".join(
                    self.all_next_skill_aggregate_skills(skill)
                    for skill in process_skill_strings(available_next_skills)
                )
                + "\n"
            )
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(prompt_part_one)
            ]
            next_i = len(prompt_part_one)
            modified_prompts_without_end.append(
                self.all_next_skill_prompt_start
                + all_next_skill_sentence
                + self.starter
                + "".join(with_mid_annotations)
                + str(next_i + 1)
                + "."
            )
        import os
        import openai
        import backoff

        openai.api_key = os.getenv("OPENAI_API_KEY")

        @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
        def generate_completion(prompts):
            try:
                return openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompts,
                    max_tokens=25,
                    stop="\n",
                    temperature=self.next_skill_temp,
                    top_p=self.next_skill_top_p,
                    logprobs=1,
                    n=num_generations,
                )
            except:
                raise openai.error.RateLimitError
            # except openai.error.ServiceUnavailableError:
            #    raise openai.error.RateLimitError
            # except openai.error.APIError:
            #    raise openai.error.RateLimitError
            # except openai.error.APIConnectionError:
            #    raise openai.error.RateLimitError
            # except requests.exceptions.ReadTimeoutError:
            #    raise openai.error.RateLimitError
            # except openai.error.Timeout:
            #    raise openai.error.RateLimitError

        batch_size = 20
        completions = []
        next_skill_preds = []
        logprobs = []
        for i in range(0, len(modified_prompts_without_end), batch_size):
            completions = generate_completion(
                modified_prompts_without_end[i : i + batch_size]
            )
            for completion in completions.choices:
                next_skill_preds.append(completion.text)
                logprobs.append(
                    torch.tensor(completion.logprobs.token_logprobs).mean(
                        -1, keepdim=True
                    )
                )
        logprobs = torch.cat(logprobs, dim=0)
        return process_skill_strings(next_skill_preds), logprobs

    def generate_next_skill_codex(
        self, first_annotations: list[list[str]], num_generations
    ):
        modified_prompts_without_end = []
        for prompt_part_one in first_annotations:
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(prompt_part_one)
            ]
            next_i = len(prompt_part_one)
            modified_prompts_without_end.append(
                self.summary_prompt_start
                + "".join(with_mid_annotations)
                + str(next_i + 1)
                + "."
            )
        import os
        import openai
        import backoff

        openai.api_key = os.getenv("OPENAI_API_KEY")

        @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
        def generate_completion(prompts):
            try:
                return openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompts,
                    max_tokens=25,
                    stop="\n",
                    temperature=self.next_skill_temp,
                    top_p=self.next_skill_top_p,
                    logprobs=1,
                    n=num_generations,
                )
            except openai.error.ServiceUnavailableError:
                raise openai.error.RateLimitError
            except openai.error.APIError:
                raise openai.error.RateLimitError
            except openai.error.APIConnectionError:
                raise openai.error.RateLimitError
            except openai.error.Timeout:
                raise openai.error.RateLimitError
            except requests.exceptions.ReadTimeout:
                raise openai.error.RateLimitError

        batch_size = 20
        completions = []
        next_skill_preds = []
        logprobs = []
        for i in range(0, len(modified_prompts_without_end), batch_size):
            completions = generate_completion(
                modified_prompts_without_end[i : i + batch_size]
            )
            for completion in completions.choices:
                next_skill_preds.append(completion.text)
                logprobs.append(
                    torch.tensor(completion.logprobs.token_logprobs).mean(
                        -1, keepdim=True
                    )
                )
        logprobs = torch.cat(logprobs, dim=0)
        return process_skill_strings(next_skill_preds), logprobs

    def _generate_hf_text(self, all_tokenized_prompts, num_generations, ret_logprobs=True):
        composite_skill_annotations = []
        all_responses = []
        # for i in range(0, len(all_tokenized_prompts.input_ids), self.llm_batch_size):
        annotation_ids = all_tokenized_prompts.input_ids[0:1].to(self.llm_gpus[0])
        annotation_attn_mask = all_tokenized_prompts.attention_mask[0:1].to(
            self.llm_gpus[0]
        )
        for i in range(0, num_generations, self.llm_batch_size):
            with self.lock:
                bad_response = True
                while bad_response:
                    responses = self.model.generate(
                        annotation_ids,
                        attention_mask=annotation_attn_mask,
                        return_dict_in_generate=True,
                        early_stopping=True,
                        max_new_tokens=self.llm_max_new_tokens,
                        do_sample=True,
                        top_p=self.next_skill_top_p,
                        temperature=self.next_skill_temp,
                        num_return_sequences=min(self.llm_batch_size, num_generations - i),
                        # num_beams=5,
                        # output_scores=True,
                        output_scores=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    new_labels = self.process_hf_generation(responses)
                    bad_response = new_labels is False
            composite_skill_annotations.extend(new_labels)
            all_responses.append(responses)
        logprobs = []
        if ret_logprobs:
            for responses in all_responses:
                if "scores" in responses:
                    logprobs.append(self._get_logprobs_hf(responses))
        if len(logprobs) > 0:
            logprobs = torch.cat(logprobs, dim=0)
        else:
            logprobs = None
        return composite_skill_annotations, logprobs

    def generate_next_skill_with_other_skills(
        self,
        all_annotation_list: list[list[str]],
        next_skill_candidates: list[list[str]],
        num_generations,
    ):
        if self.generate_next_skill_with_codex:
            return self.generate_next_skill_with_other_skills_codex(
                all_annotation_list, next_skill_candidates, num_generations
            )
        all_tokenized_prompts = self.preprocess_llm_inputs_for_choosing_next_skill_generation(
            all_annotation_list, next_skill_candidates
        )
        return self._generate_hf_text(all_tokenized_prompts, num_generations)

    def generate_next_skill(
        self, all_annotation_list: list[list[str]], num_generations
    ):
        if self.generate_next_skill_with_codex:
            return self.generate_next_skill_codex(all_annotation_list, num_generations)
        all_tokenized_prompts = self.preprocess_llm_inputs_for_logprob_generation(
            all_annotation_list
        )
        return self._generate_hf_text(all_tokenized_prompts, num_generations)

    def generate_skill_labels_with_codex(self, all_annotation_list: list[list[str]]):
        modified_prompts = []
        for primitive_annotations in all_annotation_list:
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(primitive_annotations)
            ]
            modified_prompts.append(
                self.summary_prompt_start
                + "".join(with_mid_annotations)
                + self.summary_prompt_end
            )
        import os
        import openai
        import backoff

        openai.api_key = os.getenv("OPENAI_API_KEY")

        @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
        def generate_completion(prompts):
            # convert prompts to gpt-3.5-turbo format of list of dicts 
            messages = [
                dict(role="user", content=prompt) for prompt in prompts
            ] 
            try:
                return openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    #prompt=prompts,
                    messages=messages,
                    max_tokens=25,
                    stop="\n",
                    temperature=0.5,
                )
            except:
                raise openai.error.RateLimitError
            # except openai.error.ServiceUnavailableError:
            # raise openai.error.RateLimitError
            # except openai.error.APIError:
            #     raise openai.error.RateLimitError
            # except openai.error.APIConnectionError:
            #     raise openai.error.RateLimitError
            # except openai.error.Timeout:
            #     raise openai.error.RateLimitError
            # except requests.exceptions.ReadTimeout:
            #     raise openai.error.RateLimitError

        batch_size = 1  # max of 1 prompt at a time because gpt-3.5-turbo cannot accept batched requests
        completions = []
        composite_skill_annotations = []
        for i in range(0, len(modified_prompts), batch_size):
            completions = generate_completion(modified_prompts[i : i + batch_size])
            for completion in completions.choices:
                composite_skill_annotations.append(completion.message.content)
                #composite_skill_annotations.append(completion.text)
        return process_skill_strings(composite_skill_annotations)

    def generate_skill_labels(self, all_annotation_list: list[list[str]], use_codex):
        if use_codex:
            return self.generate_skill_labels_with_codex(all_annotation_list)
        composite_skill_annotations = []
        for i in range(0, len(all_annotation_list), self.llm_batch_size):
            tokenized_prompts = self.preprocess_llm_inputs_for_summarization(
                all_annotation_list[i: i + self.llm_batch_size]
            )
            batch_annotation_ids = tokenized_prompts.input_ids
            
            batch_annotation_attn_mask = tokenized_prompts.attention_mask
            
            with self.lock:
                bad_response = True
                while bad_response:
                    responses = self.model.generate(
                        batch_annotation_ids.to(self.llm_gpus[0]),
                        attention_mask=batch_annotation_attn_mask.to(self.llm_gpus[0]),
                        return_dict_in_generate=True,
                        early_stopping=True,
                        max_new_tokens=self.llm_max_new_tokens,
                        do_sample=True,
                        top_p=self.next_skill_top_p,
                        temperature=self.summary_temp,
                        #temperature=self.next_skill_temp,
                        # num_beams=5,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    new_labels = self.process_hf_generation(responses)
                    bad_response = new_labels is False
            composite_skill_annotations.extend(new_labels)
        return composite_skill_annotations


def get_nli_logprobs(
    pred_task_names, prompt_part_ones, prompt_part_twos, nli_classifier
):
    # gets logprobs of NLI predictions for each task, where we're having 1 label per skill summary
    # prompt looks like: First, SKILL 1. Then, SKILL 2.
    skill_descriptions = []
    for i in range(len(pred_task_names)):
        prompt_part_one = prompt_part_ones[i]
        prompt_part_two = prompt_part_twos[i]
        skill_description = f"First, {prompt_part_one}. Then, {prompt_part_two}"
        skill_descriptions.append(skill_description)
    responses = nli_classifier(
        skill_descriptions, pred_task_names, multi_label=True
    )  # pred_task_names is the list of labels for NLI
    probs = [
        response["scores"][response["labels"].index(pred_task_names[i])]
        for i, response in enumerate(responses)
    ]
    logprobs = np.log(probs)
    return logprobs, skill_descriptions


# cache testing
#if __name__ == "__main__":
    #from transformers import FlaxAutoModelForCausalLM
    #from transformers import GPT2Tokenizer
    #import jax

    #opt_test_model = AutoModelForCausalLM.from_pretrained(
    #    "facebook/opt-350m", use_cache=True,
    #).to(0)
    #opt_flax_model = FlaxAutoModelForCausalLM.from_pretrained(
    #    # "facebook/opt-350m", use_cache=True, dtype=jax.numpy.float16, from_pt=True
    #    "facebook/opt-350m",
    #)
    ## opt_flax_model.params = opt_flax_model.to_fp32(opt_flax_model.params)
    ## jax.device_put(opt_flax_model, jax.devices()[0])
    #opt_test_tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
    ## opt_test_tokenizer = AutoTokenizer.from_pretrained(
    ##    "facebook/opt-350m", use_fast=False
    ## )

    #@jax.jit
    #def flax_generate(input_ids, attn_masks):
    #    output = opt_flax_model.generate(
    #        input_ids=input_ids,
    #        attention_mask=attn_masks,
    #        # input_ids=tokenized_input.input_ids,
    #        # output_attentions=True,
    #        # return_dict=True,
    #        # return_dict_in_generate=True,
    #        # use_cache=True,
    #        # past_key_values=past_key_values,
    #        max_new_tokens=20,
    #    )
    #    return output

    #opt_test_tokenizer.padding_side = "left"
    ## opt_test_tokenizer.pad_token_id = opt_test_tokenizer.eos_token_id
    ## opt_test_tokenizer.pad_token = opt_test_tokenizer.eos_token

    #input_test = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."
    #input_test2 = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum. Yas."
    #tokenized_input = opt_test_tokenizer(
    #    [input_test, input_test2], return_tensors="pt", padding=True, truncation=True
    #)
    #tokenized_input.input_ids = tokenized_input.input_ids.to(0)
    #tokenized_input.attention_mask = tokenized_input.attention_mask.to(0)

    ## past_key_values = opt_test_model(
    ##        input_ids=tokenized_input.input_ids,
    ##        attention_mask=tokenized_input.attention_mask,
    ##        output_attentions=True,
    ##        return_dict=True,
    ##        use_cache=True
    ##        ).past_key_values
    ## next_thing = "dolor"
    ## old_attention_mask = tokenized_input.attention_mask
    ## old_input = tokenized_input.input_ids
    ## tokenized_input = opt_test_tokenizer(next_thing, return_tensors="pt")
    #tokenized_input.input_ids = tokenized_input.input_ids.to(0)
    #tokenized_input.attention_mask = tokenized_input.attention_mask.to(0)
    ## tokenized_input.input_ids = tokenized_input.input_ids.to(0)[:, 1:]
    ## tokenized_input.attention_mask = tokenized_input.attention_mask.to(0)[:, 1:]
    #import time

    ## start = time.time()
    ## output = opt_test_model.generate(
    ##    input_ids=tokenized_input.input_ids,
    ##    attention_mask=tokenized_input.attention_mask,
    ##    max_new_tokens=20,
    ##    return_dict_in_generate=True,
    ## )
    ### output = opt_test_model.generate(
    ###        #input_ids=torch.cat((old_input, tokenized_input.input_ids), dim=1),
    ###        input_ids=tokenized_input.input_ids,
    ###        attention_mask=torch.cat((old_attention_mask, tokenized_input.attention_mask), dim=1),
    ###        #attention_mask=tokenized_input.attention_mask,
    ###        #output_attentions=True,
    ###        #return_dict=True,
    ###        return_dict_in_generate=True,
    ###        use_cache=True,
    ###        max_new_tokens=20,
    ###        #past_key_values=past_key_values,
    ###        past=past_key_values,
    ###        )
    ## end = time.time()
    ## print(end - start)
    ## start = time.time()
    ## output = opt_test_model.generate(
    ##    input_ids=tokenized_input.input_ids,
    ##    attention_mask=tokenized_input.attention_mask,
    ##    max_new_tokens=20,
    ##    return_dict_in_generate=True,
    ## )
    ### output = opt_test_model.generate(
    ###        #input_ids=torch.cat((old_input, tokenized_input.input_ids), dim=1),
    ###        input_ids=tokenized_input.input_ids,
    ###        attention_mask=torch.cat((old_attention_mask, tokenized_input.attention_mask), dim=1),
    ###        #attention_mask=tokenized_input.attention_mask,
    ###        #output_attentions=True,
    ###        #return_dict=True,
    ###        return_dict_in_generate=True,
    ###        use_cache=True,
    ###        max_new_tokens=20,
    ###        #past_key_values=past_key_values,
    ###        past=past_key_values,
    ###        )
    ## end = time.time()
    ## print(end - start)
    ## print(opt_test_tokenizer.batch_decode(output["sequences"]))
    ## print(opt_test_tokenizer.batch_decode([output.logits[:, -1:].argmax()]))
    ## tokenized_input.input_ids = tokenized_input.input_ids.to(0)[:, 1:]
    ## tokenized_input.attention_mask = tokenized_input.attention_mask.to(0)[:, 1:]
    ## tokenized_input = opt_test_tokenizer(input_test, return_tensors="jax")
    #opt_test_tokenizer.padding_side = "right"
    #tokenized_input = opt_test_tokenizer(
    #    [input_test, input_test2], return_tensors="jax", padding=True, truncation=True
    #)
    #import pdb

    #pdb.set_trace()
    #tokenized_input.input_ids = jax.device_put(
    #    tokenized_input.input_ids, jax.devices()[0]
    #)
    #tokenized_input.attention_mask = jax.device_put(
    #    tokenized_input.attention_mask, jax.devices()[0]
    #)
    #start = time.time()
    ## output = flax_generate(tokenized_input.input_ids, tokenized_input.attention_mask)
    #output = opt_flax_model.generate(
    #    tokenized_input.input_ids,
    #    attention_mask=tokenized_input.attention_mask,
    #    max_new_tokens=20,
    #)
    #end = time.time()
    #print(end - start)
    #start = time.time()
    ## output = flax_generate(tokenized_input.input_ids, tokenized_input.attention_mask)
    #output = opt_flax_model.generate(
    #    tokenized_input.input_ids,
    #    attention_mask=tokenized_input.attention_mask,
    #    max_new_tokens=20,
    #)
    #end = time.time()
    #print(end - start)
    #print(opt_test_tokenizer.batch_decode(output["sequences"]))
    #import pdb

    #pdb.set_trace()
    # print(opt_test_tokenizer.batch_decode([output.logits[:, -1:].argmax()]))
    # generated_tokens = opt_test_model.generate(
    #    input_ids=tokenized_input.input_ids,
    #    do_sample=False,
    #    return_dict_in_generate=True,
    #    output_hidden_states=True,
    #    max_new_tokens=0,
    #    use_cache=True,
    # )
    # generated_tokens = opt_test_model.generate(
    #    input_ids=tokenized_input.input_ids,
    #    do_sample=False,
    #    return_dict_in_generate=True,
    #    hidden_states=generated_tokens.hidden_states,
    #    max_new_tokens=50,
    # )

    # num_new_tokens = (
    #    generated_tokens.input_ids.shape[1] - tokenized_input["input_ids"].shape[1]
    # )

    # curr_input_ids = tokenized_input["input_ids"]
    # curr_attn_masks = tokenized_input["attention_mask"]
    # curr_past_key_values = None
    # for i in range(num_new_tokens):
    #    curr_output = opt_test_model(
    #        input_ids=curr_input_ids,
    #        attention_mask=curr_attn_masks,
    #        use_cache=True,
    #        past_key_values=curr_past_key_values,
    #        return_dict=True,
    #    )
    #    correct_input_id = torch.argmax(curr_output.logits, dim=-1)
    #    curr_input_ids = torch.cat(
    #        (curr_input_ids, curr_output.sequences[:, -1]), dim=1
    #    )
    #    curr_attn_masks = torch.cat(
    #        (curr_attn_masks, torch.ones(curr_output.sequences.shape[0], 1)), dim=1
    #    )
    #    if i == 0:
    #        curr_hidden_states = curr_output.hidden_states[-1]
    #    else:
    #        curr_hidden_states = (*curr_hidden_states, curr_output.hidden_states[-1])
