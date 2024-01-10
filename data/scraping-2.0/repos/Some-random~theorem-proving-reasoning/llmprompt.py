import time
import json
import random
import datetime
import logging
import os
import re

import fire
import openai
from tqdm import tqdm

from llms.inference import generate
from prompts import *


logger = logging.getLogger(__name__)

try:
    import utils

    openai.api_key = utils.openai_api_key
    openai.organization = utils.openai_edinburgh_organization
except ImportError:
    logger.warning("Could not import OpenAI credentials")
    openai.api_key = None
    openai.organization = None

seed = 2024
split_formalization_and_proof = False
use_COT = True
use_COT_in_all = True
add_comment_and_use_COT = True

divinci_config = {
    "temperature": 0,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": "\n------------",
    "max_tokens": 512,
}
gpt_config = {
    "temperature": 0,
    "top_p": 1.0,
}

if not split_formalization_and_proof:
    prompt_inputs = [
        ProofWriter_example_true_textual_input,
        ProofWriter_example_false_textual_input,
        ProofWriter_example_unknown_textual_input,
    ]
    if not use_COT_in_all:
        prompt_outputs = [
            ProofWriter_example_true_all_output,
            ProofWriter_example_false_all_output,
            ProofWriter_example_unknown_all_output,
        ]
    else:
        if not add_comment_and_use_COT:
            prompt_outputs = [
                ProofWriter_example_true_all_COT_output,
                ProofWriter_example_false_all_COT_output,
                ProofWriter_example_unknown_all_COT_output,
            ]
        else:
            prompt_outputs = [
                ProofWriter_example_true_all_COT_comment_output,
                ProofWriter_example_false_all_COT_comment_output,
                ProofWriter_example_unknown_all_COT_comment_output,
            ]
else:
    prompt_inputs = [
        ProofWriter_example_true_textual_input
        + "\n---\n"
        + ProofWriter_example_true_formalization,
        ProofWriter_example_false_textual_input
        + "\n---\n"
        + ProofWriter_example_false_formalization,
        ProofWriter_example_unknown_textual_input
        + "\n---\n"
        + ProofWriter_example_unknown_formalization,
    ]
    if not use_COT:
        prompt_outputs = [
            ProofWriter_example_true_proof,
            ProofWriter_example_false_proof,
            ProofWriter_example_unknown_proof,
        ]
    else:
        prompt_outputs = [
            ProofWriter_example_true_proof_COT,
            ProofWriter_example_false_proof_COT,
            ProofWriter_example_unknown_proof_COT,
        ]
proof_writer_answer_map = {"True": "A", "False": "B", "Unknown": "C"}


def data_generation_proofwriter(filename):
    qa_pairs = []
    # open jsonl file
    with open(filename, "r") as json_file:
        json_list = list(json_file)
        for l in json_list:
            d = json.loads(l.strip())
            theory = d.get("theory")
            questions = d.get("questions")
            for key in questions.keys():
                q = questions.get(key)
                if q.get("QDep") == 5 and q.get("answer") != "Unknown":
                    question_temp = q.get("question")
                    answer = q.get("answer")
                    real_question = theory + " Question: " + question_temp + "?"
                    qa_pairs.append((real_question, answer))
    random.seed(seed)
    random_qa_pairs = random.sample(qa_pairs, 50)
    random_qa_pairs_new = random.sample(qa_pairs, 300)
    temp_random_qa_pairs = []
    for item in random_qa_pairs_new:
        if item not in random_qa_pairs:
            temp_random_qa_pairs.append(item)
    random_qa_pairs = random_qa_pairs[:5]
    return temp_random_qa_pairs


def data_generation_proofwriter_logiclm(json_file):
    d = json.load(open(json_file, "r"))
    qa_pairs = []
    random.seed(seed)
    for item in d:
        qa_pairs.append((item["context"], item["question"], item["answer"]))
    # random_qa_pairs = random.sample(qa_pairs, 100)
    return qa_pairs


def run_llm_prompt(model, prompt_input, **config):
    if model == "text-davinci-003":
        prompt = (
            "Task Description: "
            + system_message
            + "\n\n------------"
            + "Input:\n"
            + prompt_inputs[0]
            + "- - - - - - - - - - - -"
            + prompt_outputs[0]
            + "\n------------"
            + "Input:\n"
            + prompt_inputs[1]
            + "- - - - - - - - - - - -"
            + prompt_outputs[1]
            + "\n------------"
            + "Input:\n"
            + prompt_inputs[2]
            + "- - - - - - - - - - - -"
            + prompt_outputs[2]
            + "\n------------"
            + "Input:\n"
            + prompt_input
            + "- - - - - - - - - - - -"
        )
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=divinci_config["temperature"],
            max_tokens=divinci_config["max_tokens"],
            stop=divinci_config["stop"],
            presence_penalty=divinci_config["presence_penalty"],
            frequency_penalty=divinci_config["frequency_penalty"],
            top_p=divinci_config["top_p"],
        )
        content = response["choices"][0]["text"].strip()

    elif model == "gpt-4" or model == "gpt-3.5-turbo":
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Task Description: "
                    + system_message
                    + "\nI will give you some examples",
                },
                {
                    "role": "user",
                    "content": "Input:\n"
                    + prompt_inputs[0]
                    + "\n\nOutput:\n"
                    + prompt_outputs[0],
                },
                {
                    "role": "user",
                    "content": "Input:\n"
                    + prompt_inputs[1]
                    + "\n\nOutput:\n"
                    + prompt_outputs[1],
                },
                {
                    "role": "user",
                    "content": "Input:\n"
                    + prompt_inputs[2]
                    + "\n\nOutput:\n"
                    + prompt_outputs[2],
                },
                {
                    "role": "assistant",
                    "content": "Now give me the result for this input: Input:\n"
                    + prompt_input
                    + "\n\nOutput:\n",
                },
            ],
            temperature=gpt_config["temperature"],
            top_p=gpt_config["top_p"],
        )
        content = response["choices"][0]["message"]["content"].strip()
    else:
        system_prompt = (
            f"Task Description: {system_message}\nI will give you some examples"
        )
        context_prompt = [
            f"Input:\n{prompt_inputs[ii]}\n\nOutput:\n{prompt_outputs[2]}"
            for ii in range(len(prompt_inputs))
        ]
        context_prompt.append("Now give me the result for this input:\n\nInput:")
        context_prompt = "\n".join(context_prompt)
        user_prompt = "\n\nOutput:\n"
        content = generate(
            model,
            [prompt_input],
            model_system_prompt=system_prompt,
            model_context_prompt=context_prompt,
            model_user_prompt=user_prompt,
            use_model_cache=True,
            show_progress=False,
            **config,
        )[0]
        response = None

    return content, response


def get_model_answer(model_output):
    answer_match = re.search('.*the answer is (\w+)', model_output)
    if answer_match:
        answer_key = answer_match.groups()[0]
    else:
        answer_key = model_output.strip().split("\n")[-1].split(" ")[-1]
    
    answer_key = answer_key.replace(".", "").capitalize()
    predicted_answer = proof_writer_answer_map[answer_key]
    return predicted_answer


def run_prompt(random_qa_pairs, model, **kwargs):
    # make a folder to store the outputs and config files, make sure the folder contain current timestamp and model name
    timestamp = datetime.datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
    folder_name = "ProofWriter_" + timestamp + "_" + model.split('/')[-1]
    os.mkdir(folder_name)

    # dump the prompt to a file
    write_file = open(folder_name + "/prompt.txt", "w")
    write_file.write("System message:\n")
    write_file.write(system_message + "\n\n")
    for i in range(len(prompt_inputs)):
        write_file.write("Example " + str(i + 1) + ":\n")
        write_file.write(
            "Input:\n"
            + prompt_inputs[i]
            + "\n\n"
            + "Output:\n"
            + prompt_outputs[i]
            + "\n\n"
        )
    write_file.close()

    if model == "text-davinci-003":
        config = divinci_config
    elif model.startswith("gpt"):
        config = gpt_config
    else:
        config = kwargs

    # dump the config to a json file
    with open(folder_name + "/config.json", "w") as json_file:
        json.dump(config, json_file)

    # make a json file to store the random_qa_pairs and its corresponding outputs
    for i in tqdm(range(len(random_qa_pairs))):
        try:
            qa_pair = random_qa_pairs[i]
            if not split_formalization_and_proof:
                prompt_input = (
                    "Textual context: " + qa_pair[0] + "\n" + "Question: " + qa_pair[1]
                )
            else:
                print(
                    "If you want to separate the formalization and proof process, you need to first generate the formalization then put the formalization into the prompt to generate proof. It is not supported in this project because it will generate inferior results."
                )
                exit(0)

            content, response = run_llm_prompt(model, prompt_input, **config)
            json_file = open(folder_name + "/output_" + str(i) + ".json", "w")
            predicted_answer = get_model_answer(content)

            d = {
                "input": prompt_input,
                "output": content,
                "pred_answer": predicted_answer,
                "gt_answer": qa_pair[-1],
                "problem_id": i,
            }

            if response and "usage" in response:
                d["input_tokens"] = response["usage"]["prompt_tokens"]
                d["output_tokens"] = response["usage"]["completion_tokens"]

            print(
                "This is problem: "
                + str(i)
                + ". The predicted answer is: "
                + predicted_answer
                + ", The GT answer is: "
                + str(qa_pair[-1])
            )
            json.dump(d, json_file, indent=4, ensure_ascii=False)
            json_file.close()
        except KeyError as ex:
            print("Error in problem: " + str(i) + ".")
            print("Answer key:", ex)
            print("Model output:\n", content)


def run(data_path, model="gpt-3.5-turbo", **kwargs):
    start_time = time.time()
    # select random questions from proofwriter OWA depth-5 dataset, to use it, download it from https://allenai.org/data/proofwriter
    # random_qa_pairs = data_generation_proofwriter(filename='proofwriter-dataset-V2020.12.3/OWA/depth-5/meta-test.jsonl')
    # as we're following the same setup as LogicLM, we use the same data
    proof_writer_dev_qa_pairs = data_generation_proofwriter_logiclm(data_path)
    run_prompt(proof_writer_dev_qa_pairs, model, **kwargs)
    print("Time elapsed: ", time.time() - start_time)


if __name__ == "__main__":
    fire.Fire(run)

    """To run this script:

    1. Install thefonseca/llms:
    > git clone https://github.com/thefonseca/llms.git
    > cd llms
    > pip install -e .

    2. Run inference with LLama-2 (other models based on huggingface are supported):

    > python llmprompt.py \
        --data_path ../Logic-LLM/data/ProofWriter/dev.json \
        --model llama-2-7b-chat \
        --model_checkpoint_path /path/to/hugginface/checkpoint/ \
        --model_dtype float16 \
        --max_length 512


    > python llmprompt.py \
        --data_path ../Logic-LLM/data/ProofWriter/dev.json \
        --model codellama/CodeLlama-7b-Instruct-hf \
        --model_dtype float16 \
        --max_length 512
    """