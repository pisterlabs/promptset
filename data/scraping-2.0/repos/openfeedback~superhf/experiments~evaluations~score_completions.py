"""
This is a file for running a reward model to score a text file of completions

Language model naming conventions:
    if @ is in it, what follows is a revision / branch name
    if "{N}" is in it, then the name must be author/model_name-{N} where
        we search the hub for all models of this form under author's account
"""

import argparse
import os
import json
from typing import Optional, List, Dict, Any
import random
import requests
import yaml

from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    GPT2Tokenizer,
)
import torch
from torch.utils.data import DataLoader, TensorDataset

from accelerate import Accelerator, find_executable_batch_size

from model_loading import load_eval_model_and_tokenizer
from evaluation_utils import trim_generations


from tqdm import tqdm
import wandb

from superhf.data import get_superhf_prompts
from superhf.utils import print_memory_utilization, BestOfNWrapper
from superhf.training import SuperHFTrainingArguments, SuperHFTrainer
from superhf.filtering import CompletionFilterTopK
from superhf.mocking import MockRewardModel

# Default parameters
WANDB_ENTITY_NAME = "stanfordaialignment"
WANDB_PROJECT_NAME = "rlhf-trl-v1"

# folder to be used for saving and loading test set eval data
# TEST_COMPLETIONS_DIR = "test_completions"
# TEST_SCORES_DIR = "test_scores"

# Saved conversation prompt
CONVERSATION_PROMPT = (
    "A human user sends a message, and a helpful and harmless AI assistant responds."
)
VERBOSE = False


def parse_args() -> argparse.Namespace:
    """
    This function parses the arguments for which completions to load, and
    which reward model(s) to use, if any.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./experiments/evaluations/configs/generate_completions_default.yaml",
        help="Path to the config file containing the completions to score",
    )
    parser.add_argument(
        "--completions-dir",
        type=str,
        default="test_completions",
        help="Directory name where the generations will be saved and read from,",
    )
    parser.add_argument(
        "--scores-dir",
        type=str,
        default="test_scores",
        help="The directory name where scores will be saved to.",
    )
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        yaml_args = yaml.safe_load(file)

    values_dict = {k: v["value"] for k, v in yaml_args.items()}
    values_dict["completions_dir"] = args.completions_dir
    values_dict["scores_dir"] = args.scores_dir
    return argparse.Namespace(**values_dict)


def load_completions_wandb(
    entity_name=WANDB_ENTITY_NAME, project_name=WANDB_PROJECT_NAME, run_id="0tjh64pg"
) -> tuple[list[list[str]], list[list[float]]]:
    """
    Calls the wandb api to load completions and scores from a specified run.
    It is run sequentially, and it downloads all versions of the run until it
    encounters an error.

    TODO: make this run in parallel

    Returns:
        completions: a list of lists of completions. Shape is (num_epochs, batch_size)
        scores: a list of lists of corresponding scores. Shape is (num_epochs, batch_size)
    """

    assert run_id is not None, "Must specify a run id"

    # authenticate with wandb
    wandb.login()

    # initialize the API
    api = wandb.Api()

    completions = []
    scores = []
    version_num = 0

    while True:
        # retrieve the table data
        wandb_path_to_data = os.path.join(
            entity_name,
            project_name,
            "run-" + run_id + "-game_log:v" + str(version_num),
        )
        try:
            artifact = api.artifact(wandb_path_to_data, type="run_table")
            artifact_save_path = artifact.download()
        except wandb.errors.CommError:
            break

        # retrieve the json file
        with open(
            os.path.join(artifact_save_path, "game_log.table.json"),
            "r",
            encoding="utf-8",
        ) as table_file:
            game_log = json.load(table_file)

        completions.append(
            [game_log["data"][i][1] for i in range(len(game_log["data"]))]
        )
        scores.append([game_log["data"][i][2] for i in range(len(game_log["data"]))])
        version_num += 1

    return completions, scores


def read_openai_completions(input_file: str) -> list[str]:
    """
    Read openai completions from a file.

    Args:
        input_file: The path to the file containing the answers

    Returns:
        A list of answers
    """
    # check if it's a file
    if not os.path.isfile(input_file):
        raise ValueError(
            f"Input file {input_file} does not exist. cwd is {os.getcwd()}"
        )

    with open(input_file, "r", encoding="utf-8") as file:
        completions = json.load(file)["completions"]
    return completions


def score_completions(
    starting_batch_size_rm: int,
    reward_model: PreTrainedModel,
    reward_model_tokenizer: PreTrainedTokenizer,
    completions: List[str],
    reward_model_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Scores the completions from the reward model.

    Args:
        reward_model: The reward model to use.
        reward_model_tokenizer: The tokenizer for the reward model.
        completions: The completions to score, in text format.
        reward_model_kwargs: The arguments to pass to the reward model.

    Returns:
        A list of scores (logits as torch.Tensor) for each completion.
    """
    if reward_model_kwargs is None:
        reward_model_kwargs = {}

    reward_model_tokenizer.pad_token = reward_model_tokenizer.eos_token
    tqdm.write(f"We are scoring {len(completions)} completions.")
    completions_tokenized = reward_model_tokenizer(
        completions, padding=True, truncation=True, return_tensors="pt"
    )
    completions_tokenized = completions_tokenized.to(reward_model.device)
    tqdm.write(f"Moved completions to {reward_model.device}.")
    assert reward_model.device != "cpu", "Reward model must be on GPU."

    # Create a DataLoader
    tensor_dataset = TensorDataset(
        completions_tokenized["input_ids"], completions_tokenized["attention_mask"]
    )
    data_loader = DataLoader(tensor_dataset, batch_size=starting_batch_size_rm)

    scores = []
    with torch.no_grad():
        tqdm.write("Scoring completions.")
        for batch in tqdm(data_loader, desc="scoring completion batch"):
            input_ids, attention_mask = batch
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                **reward_model_kwargs,
            }
            batch_scores = reward_model(**model_inputs)
            if not isinstance(batch_scores, torch.Tensor):
                batch_scores: torch.Tensor = batch_scores.logits
            scores.append(batch_scores)
    return torch.cat(scores), starting_batch_size_rm


def generate_completions_from_lm(
    starting_batch_size,
    language_model,
    prev_language_model,
    prev_lm_tokenizer,
    prompts_dict,
):
    """
    Given a language model and test prompts and then generates completions

    Args:
        starting_batch_size: The batch size to use for the first batch.
        language_model: The language model to use.
        prev_language_model: The previous language model to use.
        prev_lm_tokenizer: The previous language model tokenizer to use.
        prompts_dict: The prompts to use.

    Returns:
        A list of completions and cached model and tokenizer, as well as a potentially
        reduced batch size due to cuda oom.
    """
    # pylint: disable=too-many-locals
    print_memory_utilization()

    language_model_path = language_model.split("@")[0]
    try:
        revision = language_model.split("@")[1]
    except IndexError:
        revision = None
    reward_model_path = None
    if "best_of_n" in language_model_path:
        parts = language_model_path.split("_best_of_n_")
        error_msg = "_best_of_n_ should be proceeded by LM path and followed by RM path"
        assert len(parts) == 2, error_msg
        language_model_path, reward_model_path = parts[0], parts[1]
    language_model, language_tokenizer = load_eval_model_and_tokenizer(
        model_path=language_model_path,
        prev_model=prev_language_model,
        prev_tokenizer=prev_lm_tokenizer,
        revision=revision,
        model_type="language",
        tokenizer_padding_side="left",
        torch_dtype=torch.bfloat16,  # kwargs
    )
    if reward_model_path is not None:
        reward_model, reward_tokenizer = load_eval_model_and_tokenizer(
            model_path=reward_model_path,
            model_type="reward",
            torch_dtype=torch.bfloat16,
        )
        language_model = BestOfNWrapper(
            language_model=language_model,
            reward_model=reward_model,
            language_tokenizer=language_tokenizer,
            reward_tokenizer=reward_tokenizer,
        )
    print_memory_utilization()

    training_args = SuperHFTrainingArguments(
        max_new_tokens=128,
        temperature=0.7,
        kl_coefficient=-1,
        superbatch_size=1,  # we don't want to duplicate prompts
        minibatch_size_generating=starting_batch_size,
    )
    # this completion filter is not used because we do not call trainer.train
    completion_filter = CompletionFilterTopK(1)

    reward_model_train = MockRewardModel()
    # load gpt2 tokenizer
    reward_tokenizer_train = GPT2Tokenizer.from_pretrained("gpt2")

    trainer = SuperHFTrainer(
        language_model=language_model,
        reward_model_train=reward_model_train,
        reward_model_val=None,
        language_tokenizer=language_tokenizer,
        reward_tokenizer_train=reward_tokenizer_train,
        reward_tokenizer_val=None,
        completion_filter=completion_filter,
        training_args=training_args,
    )

    completions_dict = {}
    for dataset_name in tqdm(prompts_dict.keys(), desc="Datasets"):
        prompts = prompts_dict[dataset_name]
        if starting_batch_size == 0:
            starting_batch_size = len(prompts)

        completions_raw = find_executable_batch_size(
            trainer.generate_completions,
            trainer.training_args.minibatch_size_generating,
        )(prompts)

        tqdm.write(
            f"The length of completions is {len(completions_raw)} for dataset"
            f" {dataset_name}"
        )
        print_memory_utilization()
        completions_filtered = trim_generations(completions_raw)
        completions_dict[dataset_name] = completions_filtered

    return (
        completions_dict,
        language_model,
        language_tokenizer,
        trainer.training_args.minibatch_size_generating,
    )


def load_prompts_dictionary(args):
    """
    Args:
        prompt_dataset_names: the list of dataets to load
    Returns:
        A dicitonary with the dataset name: list of prompts
    """
    # Get the prompt dataset
    completions_dict = {}
    for dataset_name in args.prompt_dataset_names:
        prompts = get_superhf_prompts(
            dataset_name, split="test", max_length_chars=args.max_prompt_char_length
        )

        try:
            max_prompts_per_dataset = args.max_prompts_per_dataset
        except AttributeError:
            max_prompts_per_dataset = 0
        if max_prompts_per_dataset > 0:
            if args.randomize_prompts_subset:
                prompts = random.sample(prompts, max_prompts_per_dataset)
            else:
                prompts = prompts[:max_prompts_per_dataset]
        completions_dict[dataset_name] = prompts
    print(
        f"Loaded {len(completions_dict)} datasets each of length"
        f" {len(prompts)} ({len(completions_dict) * len(prompts)} total prompts)"
    )
    return completions_dict


def load_completions_json(completions_file):
    """
    Loads completions from a json file

    Args:
        completions_file: the json file to load

    Returns:
        A dictionary with dataset name: list of completions
    """
    with open(completions_file, "r", encoding="utf-8") as file:
        completions_dict = json.load(file)
    return completions_dict


def save_completions(completions_dict, filename):
    """
    Saves completions as a json to TEST_COMPLETIONS_DIR

    Args:
        completions_dict:

    Returns:
        The filename of where the completions were saved as a json file
    """
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(completions_dict, file)
        file.write("\n")
    tqdm.write(f"Saved completions to {filename}")
    return filename


def save_scores(scores_dict, filename) -> str:
    """
    Saves scores as a json to TEST_SCORES_DIR
    """
    tqdm.write(f"Saving scores to {filename}")
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(scores_dict, file)
        file.write("\n")
    return filename


def get_all_models(model_name: str, model_interval: tuple[int, int]) -> List[str]:
    """
    Uses the hugging face api to get all the models that match the model name,
    and that fall wihin the model interval. Assumes model_name is of the form

    Args:
        model_name: the name of the model to get
        model_interval: the first model to grab and the last one, zero indexed
                        if (0,0), then load all models

    Returns:
        A list of model names with length model_interval[1] - model_interval[0]
        or if there is no placeholder {N}, just the model name
    """
    # Define the base URL for the Hugging Face Model Hub API
    base_url = "https://huggingface.co/api/"

    author, model_base_name = model_name.split("{N}")[0].split("/")
    path_pattern = author + "/" + model_base_name

    query_params = {"author": author}
    response = requests.get(base_url + "models", params=query_params, timeout=20)
    models_list = response.json()

    valid_models = [
        model["modelId"] for model in models_list if path_pattern in model["modelId"]
    ]
    valid_models.sort()

    if model_interval[0] == 0 and model_interval[1] == 0:
        model_interval = (0, len(valid_models))

    loaded_model_names = valid_models[model_interval[0] : model_interval[1]]
    print(f"Loaded model names {loaded_model_names}")
    return loaded_model_names


def remove_extension(dir_names: List[str]) -> List[str]:
    """
    Given a list of directory names reformat them to remove the .json
    """

    dir_names = [".".join(dir_name.split(".")[:-1]) for dir_name in dir_names]
    return dir_names


def generate_all_completions(
    args: argparse.Namespace,
    language_model_names,
    test_completions_dir: str,
) -> None:
    """
    loop over all the language models and generate completions for each of them
    """
    new_language_model_names = []
    for name in language_model_names:
        if "{N}" in name:
            new_language_model_names.extend(
                get_all_models(name, args.language_model_interval)
            )
        else:
            new_language_model_names.append(name)
    language_model_names = new_language_model_names
    # print(f"Current directory is {os.getcwd()}")
    if not os.path.exists(test_completions_dir):
        os.makedirs(test_completions_dir)
    already_generated_completions = os.listdir(test_completions_dir)
    already_generated_completions = remove_extension(already_generated_completions)

    # if args.wandb_run_id is not None:
    #     # grab completions from wandb
    #     completions_batched, scores_batched = load_completions_wandb(
    #         entity_name=args.wandb_entity_name,
    #         project_name=args.wandb_project_name,
    #         run_id=args.wandb_run_id,
    #     )
    #     for batch in completions_batched:
    #         completions.extend(batch)
    #     for batch_scores in scores_batched:
    #         scores.extend(batch_scores)
    starting_batch_size_lm = args.starting_batch_size_lm
    prompts_dict = load_prompts_dictionary(args)
    prev_model = None
    prev_tokenizer = None
    for language_model_name in tqdm(language_model_names, desc="Language models"):
        #  it does, don't generate here
        if "best_of_n" in language_model_name:
            parts = language_model_name.split("_best_of_n_")
            assert len(parts) == 2, "Should be two parts LM_best_of_n_RM"
            language_model_base_name = (
                parts[0].split("/")[-1] + "_best_of_n_" + parts[1].split("/")[-1]
            )
        else:
            language_model_base_name = language_model_name.split("/")[-1]
        if language_model_base_name in already_generated_completions:
            tqdm.write(
                "Already generated completions for language model"
                f" {language_model_name}"
            )
            continue
        tqdm.write(f"Generating completions with language model {language_model_name}")
        completions_dict, prev_model, prev_tokenizer, starting_batch_size_lm = (
            generate_completions_from_lm(
                starting_batch_size=starting_batch_size_lm,
                language_model=language_model_name,
                prev_language_model=prev_model,
                prev_lm_tokenizer=prev_tokenizer,
                prompts_dict=prompts_dict,
            )
        )
        filename = os.path.join(
            test_completions_dir, f"{language_model_base_name}.json"
        )
        save_completions(completions_dict, filename)


def score_all_completions(args, script_path_dir: str, test_completions_dir: str):
    """
    Run the reward model in scoring mode
    """
    # pylint: disable=too-many-locals
    scores_dict = {}
    starting_batch_size_rm = args.starting_batch_size_rm
    # we are in scoring mode, so score completions
    accelerator = Accelerator()
    try:
        torch_dtype_lm_str = args.lm_dtype
        torch_dtype_lm = getattr(torch, torch_dtype_lm_str)
    except AttributeError:
        torch_dtype_lm = torch.bfloat16

    try:
        trim_completions = args.trim_completions
    except AttributeError:
        trim_completions = False
    reward_model_name = args.reward_model
    if "_best_of_n_" in reward_model_name:
        reward_model_name = reward_model_name.split("_best_of_n_")[-1]
    reward_model, reward_tokenizer = load_eval_model_and_tokenizer(
        reward_model_name, model_type="reward", torch_dtype=torch_dtype_lm
    )
    if not isinstance(reward_model, str):
        reward_model = accelerator.prepare(reward_model)
    print_memory_utilization()

    already_generated_completions = os.listdir(test_completions_dir)
    already_generated_completions = remove_extension(already_generated_completions)
    test_scores_dir = os.path.join(script_path_dir, args.scores_dir)
    if not os.path.exists(test_scores_dir):
        os.makedirs(test_scores_dir)
    already_generated_scores = os.listdir(test_scores_dir)
    already_generated_scores = remove_extension(already_generated_scores)
    for language_model_name in tqdm(
        already_generated_completions, desc="Scoring per language models"
    ):
        if language_model_name in already_generated_scores:
            tqdm.write(
                f"Already generated scores for language model {language_model_name}"
            )
            continue
        tqdm.write(f"Scoring completions for language model {language_model_name}")
        language_model_base_name = language_model_name.split(os.path.sep)[-1]
        completions_dict = load_completions_json(
            os.path.join(test_completions_dir, f"{language_model_base_name}.json")
        )
        for dataset_name in tqdm(args.prompt_dataset_names, desc="Datasets"):
            completions = completions_dict[dataset_name]
            if trim_completions:
                completions = trim_generations(completions)
            if (
                CONVERSATION_PROMPT not in completions[0]
                and CONVERSATION_PROMPT not in completions[10]
            ):
                tqdm.write(
                    "Adding conversation prompt to completions for dataset"
                    f" {dataset_name}"
                )
                completions = [
                    CONVERSATION_PROMPT + completion for completion in completions
                ]
            tqdm.write(
                f"Here is the first completion that we are scoriing: {completions[0]!r}"
            )

            scores, starting_batch_size_rm = find_executable_batch_size(
                score_completions, starting_batch_size_rm
            )(
                reward_model=reward_model,
                reward_model_tokenizer=reward_tokenizer,
                completions=completions,
            )
            scores_dict[dataset_name] = scores.tolist()
        filename = os.path.join(test_scores_dir, f"{language_model_base_name}.json")
        save_scores(scores_dict, filename)


def main() -> None:
    """
    Main function for running the reward model
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches
    args = parse_args()
    random.seed(0)

    try:
        language_model_names = args.language_model_names
    except AttributeError:
        language_model_names = None
    try:
        scoring_mode = args.scoring_mode
    except AttributeError:
        scoring_mode = False

    script_path_dir = os.path.dirname(os.path.abspath(__file__))
    # get all the filenames in TEST_COMPLETIONS_DIR
    test_completions_dir = os.path.join(script_path_dir, args.completions_dir)

    if language_model_names is not None and len(language_model_names) != 0:
        generate_all_completions(args, language_model_names, test_completions_dir)
    elif not scoring_mode:
        raise NotImplementedError(
            "Must specify at least a language model or wandb to load generations from,"
            " or a completions file with completions from openAI. Or be in scoring mode"
        )
    if scoring_mode:
        score_all_completions(args, script_path_dir, test_completions_dir)


if __name__ == "__main__":
    main()
