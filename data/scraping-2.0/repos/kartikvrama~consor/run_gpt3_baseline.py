"""Script to generate GPT-3 responses."""

import os
import time
import json
from pathlib import Path
from datetime import datetime

import yaml
from absl import app
from absl import flags

import openai
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from helper_data import json_scene_to_prompt


flags.DEFINE_string(
    "openai_api_key",
    None,
    help="OpenAI API key to run GPT-3 model.",
)

flags.DEFINE_string(
    "test_json_data",
    None,
    help="Path to the json test data.",
)

flags.DEFINE_string(
    "config_file_path",
    "./configs/gpt3_config.yaml",
    "Path to the GPT-3 config file.",
)

# TODO: Resume automatically instead of manual.
flags.DEFINE_string(
    "resume_from_key", None, "Key to resume evaluation from in the test data."
)

FLAGS = flags.FLAGS


@retry(wait=wait_random_exponential(min=3, max=90), stop=stop_after_attempt(25))
def openai_completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def generate_new_prompt(scene, few_shot_prompt):
    """Generates a new prompt for the given json scene.

    Args:
        scene: A json scene.
        few_shot_prompt: Pre-generated few-shot prompt to be prefixed.
    """

    test_prompt = (
        str(few_shot_prompt) + "\n\n" + json_scene_to_prompt(scene, is_example=False)
    )

    return test_prompt


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    if FLAGS.openai_api_key is None:
        raise ValueError("Please provide an OpenAI API key.")
    openai.api_key = FLAGS.openai_api_key

    rate_limit_per_minute = 20
    delay = 60.0 / rate_limit_per_minute

    datetime_obj = datetime.now()
    date_time_stamp = datetime_obj.strftime("%Y_%m_%d_%H_%M_%S")

    # Load config.
    with open(FLAGS.config_file_path, "r") as fconfig:
        config = yaml.safe_load(fconfig)

    with open(config["EVAL"]["few_shot_prompt_file_path"], "r") as fh:
        few_shot_prompt = fh.read()

    test_dataset_filepath = FLAGS.test_json_data

    with open(test_dataset_filepath, "r") as fh:
        test_data = json.load(fh)
        test_keys = list(test_data.keys())

    # only if you want to resume from a particular example
    resume_from = FLAGS.resume_from_key
    start = False

    if resume_from is None:
        start = True

    else:
        print(f"Resuming from example: {resume_from}")

    results_folder = config["EVAL"]["results_folder"]
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    with open(
        os.path.join(results_folder, f"gpt3_responses_{date_time_stamp}"), "w"
    ) as fw:
        fw.write(f"Datapath:{test_dataset_filepath}\n")

        for query_key in test_keys:
            if query_key == resume_from:
                start = True

            if not start:
                continue

            print(f"Query key: {query_key}")
            fw.write(f"{query_key}\n")

            json_query_scene = test_data[query_key]

            query_prompt = generate_new_prompt(json_query_scene, few_shot_prompt)

            fw.write("[START_PROMPT]" + query_prompt + "[END_PROMPT]")

            start_time = time.time()

            # try with exponential backoff
            query_response = openai_completion_with_backoff(
                model="text-davinci-003",
                prompt=query_prompt,
                temperature=0.7,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            print(f"This took: {time.time() - start_time}")

            query_response = query_response["choices"][0]["text"]

            fw.write("[START_RESPONSE]" + query_response + "[END_RESPONSE]\n")


if __name__ == "__main__":
    app.run(main)
