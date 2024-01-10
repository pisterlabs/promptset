import json
import os
import random
import time
import warnings
from argparse import ArgumentParser
from typing import Dict, List

import datasets
import openai
import pandas as pd
from huggingface_hub import InferenceClient
from tqdm import tqdm

import final_inference_utils
from genbench import load_task
from genbench.api import PreparationStrategy


warnings.filterwarnings("ignore", message="Unverified HTTPS request is being made to host*")

os.environ["OPENAI_API_KEY"] = "OPENAI_KEY"

os.environ["CURL_CA_BUNDLE"] = ""
folder_path = final_inference_utils.get_folder_path("cot")

# LLM Parameters
model_7b_chat = "meta-llama/Llama-2-7b-chat-hf"
model_70b_chat = "meta-llama/Llama-2-70b-chat-hf"
model_13b_chat = "meta-llama/Llama-2-13b-chat-hf"
model_vicuna_13B_v1dot5 = "lmsys/vicuna-13b-v1.5-16k"
model_starcoder_chat = "HuggingFaceH4/starchat-beta"


llm_hf_dict = {
    "llama2-7b-chat": model_7b_chat,
    "llama2-13b-chat": model_13b_chat,
    "llama2-70b-chat": model_70b_chat,
    "starcoderchat": model_starcoder_chat,
}

# Inference Types
LLAMA = "llama"


def get_input_prompt_for_ans(reasoning_prompt, reasoning):
    answer_extraction_prompt = "\nAnswer: Therefore, among A through D, the answer is ("
    return reasoning_prompt + reasoning + answer_extraction_prompt


def ltr_to_idx(ltr):
    if len(ltr) != 1:
        return -10
    return ord(ltr) - ord("A")


def get_config_and_api_key_name_from_args():
    parser = ArgumentParser()
    parser.add_argument("--ds_name", help="orqa", default="cot", required=False)
    parser.add_argument("--model_name", help="Model name", default="llama2-7b-chat", required=False)
    parser.add_argument("--n_shots", type=int, help="# of shots", default=0, required=False)
    parser.add_argument("--verbose", type=int, help="1 to debug", default=0, required=False)

    args = parser.parse_args()
    print(args.verbose)
    return args


def get_llama2_predictions(ds, ds_ans, model_name):
    cot_pipeline = InferenceClient(model=model_name).text_generation

    prediction = []
    counter = 0
    for (
        ds_i,
        ds_ans_i,
    ) in tqdm(zip(ds, ds_ans), total=len(ds_ans)):
        seq = cot_pipeline(ds_i["input"], do_sample=False, max_new_tokens=500, return_full_text=False)

        new_input_prompt = get_input_prompt_for_ans(ds_ans_i["input"], seq)

        seq = cot_pipeline(new_input_prompt, do_sample=False, max_new_tokens=1, return_full_text=False)

        llm_result = -10
        llm_str_op = seq
        if llm_str_op.strip() in {"A", "B", "C", "D"}:
            llm_result = ltr_to_idx(llm_str_op)

        # If llm_result is not in the target_options then We are assuming the result to be wrong.
        if llm_result > 3 or llm_result < 0:
            llm_result = (ltr_to_idx(ds_i["original_target"]) + 1) % len(ds_i["target_options"])
            counter += 1

        prediction.append(
            {
                "reasoning_extraction_prompt": ds_i["input"],
                "answer_extraction_prompt": str(new_input_prompt),
                "target": llm_result,
            }
        )

    print(f"{counter}times LLM genrated answer that was not in the target options")

    return prediction


def get_openai_chat_prediction(ds, ds_ans, model_name, sleep_time):
    # First initialize the large language model
    # https://platform.openai.com/docs/api-reference/chat/create
    openai.api_key = os.environ["OPENAI_API_KEY"]

    options = ["A", "B", "C", "D"]
    prediction = []
    ds_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=ds))

    for (
        ds_i,
        ds_ans_i,
    ) in tqdm(zip(ds_dataset, ds_ans), total=len(ds_ans)):
        output = openai.ChatCompletion.create(
            temperature=0,
            n=1,
            model=model_name,
            max_tokens=500,
            messages=[{"role": "user", "content": ds_i["input"]}],
        )["choices"]

        reasoning = output[0]["message"]["content"]

        new_input_prompt = get_input_prompt_for_ans(ds_ans_i["input"], reasoning)

        output = openai.ChatCompletion.create(
            temperature=0,
            n=1,
            model=model_name,
            max_tokens=1,
            messages=[{"role": "user", "content": new_input_prompt}],
        )["choices"]

        # Out of range number that would be counted as incorrect
        final_prediction = 9

        for output_i in output:
            pred_str = output_i["message"]["content"]
            if pred_str in options:
                final_prediction = ltr_to_idx(str(pred_str))
                break

        prediction.append(
            {
                "reasoning_extraction_prompt": ds_i["input"],
                "answer_extraction_prompt": str(new_input_prompt),
                "target": final_prediction,
            }
        )
        # GPT-4 timeout error
        time.sleep(sleep_time)

    return prediction


def get_openai_prediction(ds, ds_ans, model_name, sleep_time):
    options = ["A", "B", "C", "D"]
    # First initialize the large language model
    openai.api_key = os.environ["OPENAI_API_KEY"]

    prediction = []
    ds_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=ds))

    for ds_i, ds_ans_i in tqdm(zip(ds_dataset, ds_ans), total=len(ds_ans)):
        reasoning = openai.Completion.create(
            temperature=0, logprobs=1, engine=model_name, max_tokens=500, prompt=ds_i["input"]
        )["choices"][0]["text"]

        new_input_prompt = get_input_prompt_for_ans(ds_ans_i["input"], reasoning)

        output = openai.Completion.create(
            temperature=0,
            logprobs=1,
            engine=model_name,
            max_tokens=1,  # we only want one letter
            prompt=new_input_prompt,
        )["choices"][0]["logprobs"]["top_logprobs"][0]

        # Out of range number that would be counted as incorrect
        final_prediction = 9
        for output_i in output:
            if output_i in options:
                final_prediction = ltr_to_idx(str(output_i))
                break

        prediction.append(
            {
                "reasoning_extraction_prompt": ds_i["input"],
                "answer_extraction_prompt": str(new_input_prompt),
                "target": final_prediction,
            }
        )

        time.sleep(sleep_time)

    return prediction


def generate_llm_prediction(ds, ds_ans, type, **kwargs) -> List[Dict]:
    """
    Return a list of dicts in the following format
    [{"target": answer1}, {"target": answer2}, ...]
    The list should be in the exact same order as the dataset (ds) passed.
    """

    if type in llm_hf_dict:
        prediction = get_llama2_predictions(ds, ds_ans, llm_hf_dict[type])
    elif type == "gpt-4":
        prediction = get_openai_chat_prediction(ds, ds_ans, "gpt-4-0613", 30)
    elif type == "gpt-3.5":
        prediction = get_openai_chat_prediction(ds, ds_ans, "gpt-3.5-turbo-0613", 15)
    elif type == "text-davinci-003":
        prediction = get_openai_prediction(ds, ds_ans, "text-davinci-003", 15)
    elif type == "gpt-3":
        prediction = get_openai_prediction(ds, ds_ans, "davinci", 15)
    else:
        prediction = [{"target": random.randint(0, 3)} for i in range(len(ds))]

    file_model_name = type.replace("/", "_")
    file_model_name = file_model_name.replace(".", "dot")
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/{file_model_name}_{str(time.time())}.json", "w+") as outfile:
        outfile.write(json.dumps(prediction, indent=4))

    return prediction


if __name__ == "__main__":
    config = get_config_and_api_key_name_from_args()
    verbose = config.verbose
    task = load_task("operationsresearchqa:" + config.ds_name)
    ds = task.get_prepared_datasets(
        PreparationStrategy.PROMPT_BASED_TESTING, shot_list=[config.n_shots], reasoning_extraction=True
    )[config.n_shots]

    task = load_task("operationsresearchqa:" + config.ds_name)
    ds_ans = task.get_prepared_datasets(
        PreparationStrategy.PROMPT_BASED_TESTING, shot_list=[config.n_shots], reasoning_extraction=False
    )[config.n_shots]

    if verbose:
        print(ds)

    if verbose:
        for i, (ds_i, ds_ans_i) in enumerate(zip(ds, ds_ans)):
            print(i)
            print(json.dumps(ds_i, indent=4))
            print(json.dumps(ds_ans_i, indent=4))
            input("next")

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("Running model", config.model_name)
    llm_output = generate_llm_prediction(ds, ds_ans, config.model_name)
    result = task.evaluate_predictions(predictions=llm_output, gold=ds)
    print("Score from experiment", result["hf_accuracy__accuracy"])
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
