import os
import sys
import string
from pprint import pprint

import hydra.utils
import openai

from src.text_generator.prompt_generator import generate_prompt_list


def gpt3(cfg, prompt):
    response = openai.Completion.create(
        prompt=prompt,
        engine=cfg.engine,
        max_tokens=cfg.response_length,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        frequency_penalty=cfg.frequency_penalty,
        presence_penalty=cfg.presence_penalty,
        stop=[". ", "\n"],  # cfg.stop_seq,
    )
    answer = response.choices[0]["text"]
    return answer


def sample_multiple_sentences_for_prompt(cfg, prompt):
    answers = []
    for i in range(cfg.num_return_sequences):
        answer = gpt3(cfg, prompt)
        print(answer)
        answers.append(answer)
    return answers


def sample_for_list_of_prompts(cfg, prompt_list_file, demo, output_dir, trigger):
    with open(prompt_list_file, "r") as f:
        prompts = [line.strip("\n") for line in f.readlines()]
    print(f"Read {len(prompts)} prompts from text file.")

    for prompt in prompts:
        print(prompt)
        sample_outputs = sample_multiple_sentences_for_prompt(cfg, prompt)
        save_outputs(output_dir, sample_outputs, demo, prompt, trigger)


def save_outputs(output_dir, sample_outputs, demo, prompt, trigger):
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"{demo}_texts.txt")
    print(f"Storing at {output_dir}")
    with open(outfile, "a") as f:
        for i, sample_output in enumerate(sample_outputs):
            if trigger:
                prompt = prompt.replace(trigger, "")
            full_sentence = prompt + sample_output + "."
            print(f"{i}: " f"{full_sentence} ")
            f.write(full_sentence + "\n")


def generate_gpt3_texts(cfg):
    # Login to GPT-3 API
    print("Enter your OpenAI authentification key.")
    auth_key_input = input()
    if len(auth_key_input) == 0:
        print(
            "ABORTING - You entered an empty string. Run script again and enter your "
            "authentification key for the GPT-3 API."
        )
        sys.exit()
    openai.api_key = auth_key_input
    # if openai.api_key is None:
    #    print("ABORTING - Your authentification seems to be incorrect.")
    #    sys.exit()

    # Check config before starting generation
    gpt3_cfg = cfg.gpt
    print(
        "Please check the following settings before starting text generation (this is a precaution "
        "since GPT-3 credits are limited)."
    )
    print("GPT-3 settings")
    pprint(gpt3_cfg)
    print("Generate mode settings")
    pprint(cfg.run_mode)
    print("Is this fine? Enter Y to confirm.")
    confirmed = input()
    if confirmed != "Y":
        print("ABORTING")
        sys.exit()

    for demo in cfg.run_mode.demographics:
        if cfg.run_mode.trigger:
            name = (
                f"{demo}_prompts_"
                f"{cfg.run_mode.trigger.translate(str.maketrans('', '', string.punctuation))}_{cfg.language}.txt"
            )
        else:
            name = f"{demo}_prompts_{cfg.language}.txt"

        prompt_dir = hydra.utils.to_absolute_path(cfg.run_mode.prompt_dir)
        output_dir = hydra.utils.to_absolute_path(cfg.run_mode.output_dir)

        file_name = os.path.join(prompt_dir, name)
        if not os.path.isfile(file_name):
            generate_prompt_list(prompt_dir, demo, cfg.run_mode.trigger, file_name)
        print(f"Sampling for {demo}")
        sample_for_list_of_prompts(
            cfg.gpt,
            file_name,
            demo,
            output_dir,
            cfg.run_mode.trigger,
        )
