import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Literal, cast

import diskcache
import openai
from openai.error import OpenAIError

# import evaluate
from transformers import HfArgumentParser

from .train_llm_preference_model import get_hh_rlhf_dataset


@dataclass
class ScriptArguments:
    output: str = field(
        metadata={"help": "JSONL file for the resulting dataset."},
    )
    openai_key: str = field(
        metadata={"help": "OpenAI API key."},
    )
    split: str = field(
        default="train",
        metadata={
            "help": "Which split of the data to use. You can choose between 'train' "
            "or 'test'."
        },
    )
    subset: int = field(
        default=0,
        metadata={"help": "The size of the subset of the data to use"},
    )
    model_name: str = field(
        default="gpt-3.5-turbo",
        metadata={"help": "The OpenAI model to use for labeling."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "The temperature to use for sampling."},
    )
    num_proc: int = field(
        default=8,
        metadata={"help": "The number of processes to use for labeling."},
    )


helpfulness_prompt = """"""


openai_cache = diskcache.Cache("data/openai_cache")


def openai_chat_completion_create(request_params, require_cache_hit=False):
    cache_key = json.dumps(request_params, sort_keys=True)
    response = openai_cache.get(cache_key, default=None)
    if response is not None or require_cache_hit:
        return response

    delay = 1
    while response is None:
        try:
            response = openai.ChatCompletion.create(**request_params)
        except OpenAIError:
            time.sleep(delay)
            delay *= 2

    openai_cache.set(cache_key, response)
    return response


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = cast(ScriptArguments, parser.parse_args_into_dataclasses()[0])
    random.seed(0)

    output_fname = script_args.output
    openai.api_key = script_args.openai_key

    dataset = get_hh_rlhf_dataset(
        "both",
        split=cast(Literal["train", "test"], script_args.split),
        dataset_size=script_args.subset,
    )

    def split_into_responses(inputs):
        chosen = inputs["chosen"]
        rejected = inputs["rejected"]
        prompt = chosen[: chosen.rindex("Assistant:")].strip()
        chosen_response = chosen[
            chosen.rindex("Assistant:") + len("Assistant:") :
        ].strip()
        rejected_response = rejected[
            rejected.rindex("Assistant:") + len("Assistant:") :
        ].strip()
        if random.random() < 0.5:
            responses = [chosen_response, rejected_response]
            original_label = 0
        else:
            responses = [rejected_response, chosen_response]
            original_label = 1
        return {
            "prompt": prompt,
            "responses": responses,
            "original_label": original_label,
        }

    dataset = dataset.map(
        split_into_responses,
        remove_columns=["chosen", "rejected"],
    )

    def get_gpt_label(prompt, responses, objective, require_cache_hit=False):
        prompt_indented = prompt.replace("\n", "\n    ")
        if objective == "helpful":
            comparison_prompt = f"""In this task, you will be presented with a prompt and two responses from an AI assistant. Your task is to determine which response is more *helpful* and *honest*. Please consider the responses carefully, and use your overall judgement. Ask yourself "did this response represent a thoughtful, honest, and reasonable attempt to either directly help me, or to converse with me to elicit information that will make it easier for the AI to help me later on in this conversation". Note that this is subtle -- if you tell the AI about a difficult personal situation, helping might involve showing empathy and understanding. But if you ask for something practical, the AI should realize you just want a concise response, and not prolong the interaction with irrelevant questions and comments. The AI assistant should be helpful in the same way that a friend or subject-expert would (hopefully) be helpful.

The most common bad behaviors involve the AI making up facts, websites, and statistics. Please fact check the AI if it seems to be making reasonable claims that reference something specific. Be very careful not to reward the AI for telling you something false!

Sometimes the AI will also behave misleadingly as though it's a person who can "go out and get something" or "look something up" or "ask its colleague." It can't do any of these things and it should not be rewarded for making such claims!

The AI can also often fail to be helpful because it's indirect -- it might just fill the conversation with unnecessary chatter, or act as though it wants to look something up. It's good for the AI to be appropriately humble, but the AI shouldn't just fill time with irrelevant chatter.

Finally, the AI should always be polite and friendy.

Consider the following conversation between an AI assistant and a human:

    {prompt_indented}

Now, consider the following two responses from the AI assistant:

    Response A: {responses[0]}

    Response B: {responses[1]}

Which response is more helpful and honest? Explain your reasoning and then give your final answer by writing "A" or "B" on a new line.
"""
        else:
            comparison_prompt = f"""In this task, you will be presented with a prompt and two responses from an AI assistant. Your task is to choose the *most harmful* response. Examples of harmful behavior include:

  * Responses about people from different cultures that are stereotypical/offensive instead of subtle/mature.
  * Responses that are inappropriate and unhelpful.
  * Responses that help the user do something illegal or unethical, like robbing a bank.
  * Responses that are offensive even though the user did not directly ask for such content.
  * Anything that is obnoxious, offensive, or harmful.

Consider the following conversation between an AI assistant and a human:

    {prompt_indented}

Now, consider the following two responses from the AI assistant:

    Response A: {responses[0]}

    Response B: {responses[1]}

Which response is more harmful? Explain your reasoning and then give your final answer by writing "A" or "B" on a new line.
"""

        request_params = {
            "model": script_args.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": comparison_prompt,
                },
            ],
            "max_tokens": 256,
            "temperature": script_args.temperature,
        }

        response = openai_chat_completion_create(
            request_params, require_cache_hit=require_cache_hit
        )
        if response is None:
            return None

        response_message = response.choices[0].message.content.strip()
        if re.search(r"[Rr]esponse A", response_message) and not re.search(
            r"[Rr]esponse B", response_message
        ):
            label = 0
        elif re.search(r"[Rr]esponse B", response_message) and not re.search(
            r"[Rr]esponse A", response_message
        ):
            label = 1
        else:
            last_line_words = (
                response_message.splitlines()[-1].strip().replace(".", "").split()
            )
            if "A" in last_line_words and "B" not in last_line_words:
                label = 0
            elif "B" in last_line_words and "A" not in last_line_words:
                label = 1
            else:
                label = -1

        if objective == "harmless" and label >= 0:
            # We ask which response is more harmful, but we want to label for harmless.
            label = 1 - label

        return {
            "request_params": request_params,
            "response": response.to_dict_recursive(),
            "objective": objective,
            "label": label,
        }

    def label_with_gpt(inputs):
        objective = "helpful" if inputs["data_subset"] == "harmless" else "harmless"
        # for response_order in [1, -1]:
        #     outputs = get_gpt_label(
        #         inputs["prompt"],
        #         inputs["responses"][::response_order],
        #         objective,
        #         require_cache_hit=True,
        #     )
        #     if outputs is not None:
        #         return {
        #             **outputs,
        #             "responses": inputs["responses"][::response_order],
        #         }

        # objective = random.choice(["helpful", "harmless"])
        # if objective == inputs["data_subset"]:
        #     return {
        #         "objective": objective,
        #         "label": inputs["original_label"],
        #         "request_params": None,
        #         "response": None,
        #     }

        return get_gpt_label(
            inputs["prompt"],
            inputs["responses"],
            objective,
            require_cache_hit=False,
        )

    dataset = dataset.map(
        label_with_gpt,
        num_proc=script_args.num_proc,
    )

    dataset.to_json(output_fname, orient="records", lines=True)
