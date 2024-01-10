import re

# Insert code for taking arguments from command line
import argparse
import openai
import os
import json
import asyncio
from api_request_parallel_processor import process_api_requests_from_file
from utils.args import add_args
from utils.misc_utils import authenticate
import random
import pandas as pd
from utils.templates import SCORING_PROMPTS, SYSTEM_PROMPTS, CLASSIFICATION_TASKS

def score_generation(args, generations, category, category_dataset, scoring_prompt):
    # score each element in category_dataset with every generation
    # generate a pandas dataframe with the scores
    # return the dataframe
    # get the prompt
    task_instruction = category_dataset[0]["task_instruction"]
    header_prompt = f"Good Job, we will now present the output generation from the language model. \n"
    # system prompt
    system_prompt = SYSTEM_PROMPTS[args.task_name]
    # generate scores for all prompts

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    generated_samples = []

    num_api_epochs = 0
    failed_to_match = 0

    id_to_generation_index = {g["id"]: i for i, g in enumerate(generations)}
    cur_generations = generations

    while num_api_epochs < 30 and len(cur_generations) > 0:
        # generate prompts for all categories and generations
        failed_to_match = 0
        prompt_file_path = os.path.join(
            args.output_dir, f"prompt_score_{category}_{num_api_epochs}.jsonl"
        )
        metadata_file_path = os.path.join(
            args.output_dir, f"metadata_score_{category}_{num_api_epochs}.jsonl"
        )
        response_file_path = os.path.join(
            args.output_dir, f"response_score_{category}_{num_api_epochs}.jsonl"
        )

        if args.delete_old_responses:
            if os.path.exists(response_file_path):
                os.remove(response_file_path)

        failed_to_match = 0
        all_prompts = []
        for generation in cur_generations:
            category_prompt = "\n".join([c[category] for c in category_dataset])
            # don't include the generation in the prompt for classification tasks
            if (
                args.task_name in CLASSIFICATION_TASKS
                or not args.use_output_for_scoring
            ):
                prompt = (
                    header_prompt
                    + scoring_prompt
                    + f"\n Make sure to score all {len(category_dataset)} {category} \n"
                    + f"{category}:\n{category_prompt}"
                )
            else:
                prompt = (
                    header_prompt
                    + scoring_prompt
                    + f"\n Make sure to score all {len(category_dataset)} {category} \n"
                    + f"{category}:\n{category_prompt} \n Output: {generation[args.generation_field]} \n"
                )
            all_prompts.append(
                {
                    "prompt": prompt,
                    "generation": generation,
                    "category_type": category,
                    "id": generation["id"],
                }
            )
        with open(prompt_file_path, "w") as f, open(
            metadata_file_path, "w"
        ) as metadata_f:
            for prompt in all_prompts:
                cur_input = prompt["generation"][args.input_field]
                formatted_request = {
                    "model": "gpt-3.5-turbo-16k",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"This is the input text: {cur_input} \n. This is the task instruction given to a language model: {task_instruction} \n",
                        },
                        {
                            "role": "user",
                            "content": f"Please understand and note the above input and the task instruction.",
                        },
                        {"role": "user", "content": prompt["prompt"]},
                    ],
                    "temperature": args.temperature,
                    "max_tokens": 1000,
                    "top_p": args.top_p,
                    "frequency_penalty": args.frequency_penalty,
                    "presence_penalty": args.presence_penalty,
                }
                metadata = {
                    "generation": prompt["generation"][args.generation_field],
                    "id": prompt["id"],
                }
                f.write(json.dumps(formatted_request))
                f.write("\n")
                metadata_f.write(json.dumps(metadata))
                metadata_f.write("\n")
        request_url = "https://api.openai.com/v1/chat/completions"

        # Make API calls
        asyncio.run(
            process_api_requests_from_file(
                prompt_file_path,
                response_file_path,
                request_url,
                args.api_key,
                args.max_requests_per_minute,
                args.max_tokens_per_minute,
                "cl100k_base",
                args.max_attempts,
                args.logging_level,
                metadata_file_path,
            )
        )
        # process the responses and save them in the data directory
        error_generations = []
        regex_template = r".*Score:\s*(?P<score>\d+)(?P<explanation>[\s|\S]*)"
        # for failed examples, refire the api call
        with open(response_file_path, "r") as f:
            api_responses = [json.loads(line) for line in f]
            for api_response in api_responses:
                metadata = api_response[2]
                api_response = api_response[1]
                if "error" in api_response or "choices" not in api_response:
                    print("Failed to generate response")
                    failed_to_match += 1
                    error_generations.append(
                        generations[id_to_generation_index[metadata["id"]]]
                    )
                    continue
                # parse the response with regex filtering
                response_text = api_response["choices"][0]["message"]["content"]
                response_scores = response_text.split("\n")
                # remove empty scores
                response_scores = [
                    s
                    for s in response_scores
                    if s and "score:" in s.lower() and "evidence" in s.lower()
                ]
                if len(response_scores) != len(category_dataset):
                    print("Failed to match example: {}".format(response_text))
                    print(
                        f"Number of scores {len(response_scores)} does not match number of category items {len(category_dataset)}"
                    )
                    failed_to_match += 1
                    error_generations.append(
                        generations[id_to_generation_index[metadata["id"]]]
                    )
                    continue
                explanation = ""
                cur_example_scores = []
                for ex_id, ex in enumerate(response_scores):
                    match = re.match(regex_template, ex)
                    if match and "score" in match.groupdict():
                        score = match.group("score").strip()
                        if "explanation" in match.groupdict():
                            explanation = match.group("explanation").strip()
                    else:
                        print("Failed to match example: {}".format(response_text))
                        error_generations.append(
                            generations[id_to_generation_index[metadata["id"]]]
                        )
                        failed_to_match += 1
                        break
                    output_example = {
                        "generation": metadata["generation"],
                        category: category_dataset[ex_id][category],
                        "score": score,
                        "id": metadata["id"],
                        "explanation": explanation,
                    }
                    cur_example_scores.append(output_example)
                if len(cur_example_scores) == len(category_dataset):
                    generated_samples.extend(cur_example_scores)
            cur_generations = error_generations
            num_api_epochs += 1
            print("Failed to match {} examples".format(failed_to_match))
            print("Retrying with {} examples".format(len(cur_generations)))
            print("Number of api epochs: {}".format(num_api_epochs))

    print("Finished all epochs. Failed to match {} examples".format(failed_to_match))
    # deal with the remaining examples
    assert len(cur_generations) == 0, "All examples could not be processed"

    output_df = pd.DataFrame.from_dict(generated_samples)
    return output_df


# input json files and extract the generation_field
# score the generation_field;
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_field",
        type=str,
        help="The key in input file for input text",
    )
    parser.add_argument(
        "--generation_field",
        type=str,
        help="The key in input file for generations",
    )
    parser.add_argument(
        "--generation_file",
        type=str,
        help="The generation file path, expecting a jsonl file",
    )
    parser.add_argument(
        "--subtask_file",
        default="",
        type=str,
        help="The subtask file path, expecting a jsonl file",
    )
    parser.add_argument(
        "--domain_file",
        default="",
        type=str,
        help="The domain file path, expecting a jsonl file",
    )
    parser.add_argument(
        "--use_output_for_scoring",
        default=1,
        type=int,
        help="Whether to use the output for scoring",
    )
    parser = add_args(parser)
    args = parser.parse_args()
    api_key = authenticate(args)
    args.api_key = api_key
    # load a jsonl file
    with open(args.generation_file, "r") as f:
        generations = [json.loads(line) for line in f]
    categories = ["subtask", "domain"]
    for category in categories:
        scoring_prompt = SCORING_PROMPTS[args.task_name][category]
        # load options for the category
        if category == "subtask":
            category_file = args.subtask_file
        elif category == "domain":
            category_file = args.domain_file
        # load category file jsonl
        with open(category_file, "r") as f:
            category_dataset = [json.loads(line) for line in f]
        score_df = score_generation(
            args, generations, category, category_dataset, scoring_prompt
        )
        # store the scores in args.output_dir
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        output_score_file = os.path.join(args.output_dir, f"{category}_scores.csv")
        score_df.to_csv(output_score_file)


if __name__ == "__main__":
    main()
