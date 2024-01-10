import typing as t
import requests
import numpy as np
import math
import time
from functools import partial
import random
from pathlib import Path
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import glob
from tqdm import tqdm
import openai
import pandas as pd
import hashlib

from tinystories import make_filter
import tinystories


INSTRUCT_MODELS = {"gpt-3.5-turbo-instruct"}


FEATURE_TO_PROMPT_QUESTION = {
    "twist": "Does the story have a twist?",
}

ZERO_SHOT_NO_REASONING_PROMPT = """You will be presented with a story written for children.
Your task is to answer the question: {question} Answer only Yes or No.

{story}

{question}"""

SHOT_PROMPT = """
<story>
{story_1}
</story>
<reasoning>{summary_1}</reasoning>
<answer>{answer_1}</answer>
"""

FEW_SHOT_PROMPT = """You will be presented with a story written for children.
Your task is to answer the question: {{question}}
Please first reason step-by-step to work out the answer to the question. Enclose your reasoning in <reasoning> tags.
After your reasoning has finished, answer the question \"{{question}}\", answering only Yes or No, and surrounding your answer with <answer> tags.
Make sure your you write our your answer inside <answer> tags on a new line after completing your reasoning. Your response must include <answer> tags.
Here are several examples:
{shots}

Here is the story to answer the question about:
<story>
{{story}}
</story>
"""

SYSTEM_PROMPT = """You are a helpful assistant, whose role is to answer questions about children's stories."""

CACHE_DIR = "cache"


def generate_story_classification_dataset(
    feature: str, dataset_size: int = 10000, use_cache: bool = True, dataset_start_index: int = 0
) -> t.Tuple[pd.DataFrame, ...]:
    """Generates a dataset of stories and their labels for a given feature."""
    if use_cache and os.path.exists(tinystories.DATA_CACHE_DIR / f"stories_{feature}.csv"):
        print("Loading cached story data...")
        stories_feature = pd.read_csv(tinystories.DATA_CACHE_DIR / f"stories_{feature}.csv")
        stories_no_feature = pd.read_csv(tinystories.DATA_CACHE_DIR / f"stories_no_{feature}.csv")
    else:
        filter_fn = make_filter(f"features={feature}")
        filter_fn_reverse = make_filter(f"features!={feature}")

        print("Loading raw story data...")
        data_dir = os.path.join(tinystories.DATA_CACHE_DIR, "TinyStories_all_data")
        shard_filenames = list(sorted(glob.glob(os.path.join(data_dir, "*.json"))))
        data = []
        for shard_filename in tqdm(shard_filenames, desc="Loading shards"):
            with open(shard_filename, "r") as f:
                shard = json.load(f)
            data.extend(shard)
            if len(data) > dataset_size * 2:
                print("Loaded enough data")
                break

        # Filter using filtering function
        print("Filtering stories...")
        data_feature: list = list(filter(filter_fn, tqdm(data, desc="Filtering stories with feature")))
        data_no_feature: list = list(
            filter(filter_fn_reverse, tqdm(data, desc="Filtering stories without feature"))
        )

        stories_feature = pd.DataFrame(data_feature)
        stories_no_feature = pd.DataFrame(data_no_feature)

        stories_feature["label"] = 1.0
        stories_no_feature["label"] = 0.0

        if use_cache:
            # save to cache
            print("Saving story data to cache...")
            stories_feature.to_csv(tinystories.DATA_CACHE_DIR / f"stories_{feature}.csv", index=False)
            stories_no_feature.to_csv(tinystories.DATA_CACHE_DIR / f"stories_no_{feature}.csv", index=False)

    # adjust dataset size - take first n rows
    print("Adjusting dataset size...")
    stories_feature = stories_feature[dataset_start_index : dataset_size + dataset_start_index]
    stories_no_feature = stories_no_feature[dataset_start_index : dataset_size + dataset_start_index]

    # Create full dataset
    stories = pd.concat([stories_feature, stories_no_feature], ignore_index=True)

    # create train test splits deterministically
    print("Creating train test splits...")
    train = stories.sample(frac=0.8, random_state=0)
    test = stories.drop(train.index)

    return train, test


def generate_cache_id(prompt, model) -> str:
    """Generate a unique cache identifier for a story and feature combination."""
    return hashlib.md5((model + prompt).encode()).hexdigest()


def get_score_for_story(
    story: str,
    feature: str,
    prompt: str,
    model: str = "gpt-3.5-turbo",
) -> float:
    """Queries the OpenAI API for the score of a story."""
    question = FEATURE_TO_PROMPT_QUESTION[feature]
    input = prompt.format(story=story.strip(), question=question)

    cache_id = generate_cache_id(input, model)
    cache_file_path = os.path.join(CACHE_DIR, cache_id)

    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as cache_file:
            return float(cache_file.read().split("\n")[0])

    if model in INSTRUCT_MODELS:
        response = openai.Completion.create(
            model=model,
            prompt=SYSTEM_PROMPT + "\n\n" + input,
            temperature=0.0,
            max_tokens=200,
        )
        content = response.choices[0].text
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        content = response.choices[0].to_dict_recursive()["message"]["content"]
    question_answer = content.split(question)[-1].strip().lower()
    if "yes" in question_answer:
        score = 1.0
    elif "no" in question_answer:
        score = 0.0
    else:
        print(f"Invalid response: {content}")
        print(prompt.format(story=story, question=question))
        score = 0.0
        # raise ValueError(f"Invalid response: {content}")

    with open(cache_file_path, "w") as cache_file:
        cache_file.write(f"{score}\n{content}")
    return score


def make_shots(stories: pd.DataFrame, n_shots: int = 0, from_cache=False) -> str:
    if from_cache:
        cached_shots = open("shots.txt", "r").read().split("\n\n")
        return "\n\n".join(cached_shots[: n_shots * 2])
    story_1_idxs = stories[stories["label"] == 1.0].index[:n_shots]
    story_2_idxs = stories[stories["label"] == 0.0].index[:n_shots]
    idxs = list(story_1_idxs) + list(story_2_idxs)
    shots = []
    for idx in idxs:
        story_1 = stories.loc[idx, "story"]
        summary_1 = stories.loc[idx, "summary"]
        answer_1 = "Yes" if stories.loc[idx, "label"] == 1.0 else "No"
        shot = SHOT_PROMPT.format(
            story_1=story_1.strip(), answer_1=answer_1.strip(), summary_1=summary_1.strip()
        )
        shots.append(shot)
    # randomly shuffle shots
    random.shuffle(shots)
    stories.drop(idxs, inplace=True)
    return "".join(shots)


def process_stories_chunk(
    stories: t.List[str],
    function_id: int,
    func: t.Callable,
) -> t.Tuple[t.List[float], int, tqdm]:
    """Process a chunk of stories."""
    result = []
    for story in (pbar := tqdm(stories, position=function_id, desc="Processing Stories")):
        while True:
            try:
                result.append(func(story))
                break
            except openai.error.RateLimitError as e:
                rate_limit_type = "tokens" if "tokens per min" in str(e) else "requests"
                pbar.set_description_str(f"Rate limited on {rate_limit_type}, sleeping...")
                time.sleep(2)
            except openai.error.ServiceUnavailableError:
                pbar.set_description_str("Service unavailable, sleeping...")
                time.sleep(5)
            except (requests.exceptions.ReadTimeout, openai.error.Timeout):
                pbar.set_description_str("Read timeout, sleeping...")
                time.sleep(10)
            except Exception as e:
                pbar.set_description_str(f"Other error: {e}, sleeping...")
                time.sleep(50)
        pbar.set_description_str("Processing Stories")

    return result, function_id, pbar


def get_score_for_stories(
    stories: pd.DataFrame,
    feature: str,
    prompt: t.Optional[str] = None,
    few_shot: int = 0,
    parallelize: bool = False,
    model: str = "gpt-3.5-turbo",
) -> np.array:
    """Queries the OpenAI API for the scores of a list of stories."""
    if few_shot:
        print("Formatting few-shot prompt")
        # get first story of true or false label, remove from stories list and use as prompt
        # they might not be the first two stories
        shots = make_shots(stories, n_shots=few_shot, from_cache=True)
        prompt = FEW_SHOT_PROMPT.format(shots=shots)
    elif prompt is None:
        prompt = ZERO_SHOT_NO_REASONING_PROMPT

    print("Querying OpenAI API...")
    func = partial(get_score_for_story, feature=feature, prompt=prompt, model=model)
    if parallelize:
        chunk_func = partial(process_stories_chunk, func=func)
        n_procs = 10
        K = math.ceil(len(stories) / n_procs)
        chunk_scores: t.List[t.List[float]] = [[] for _ in range(n_procs + 2)]
        stories_list = stories["story"].tolist()
        with ThreadPoolExecutor() as executor:
            # chunk_scores_id = list(
            #     executor.map(
            #         chunk_func,
            #         [stories_list[i * K : (i + 1) * K] for i in range(0, n_procs)],
            #         [i for i in range(1, n_procs + 1)],
            #     )
            # )

            # for chunk_score, i in chunk_scores_id:
            #     chunk_scores[i] = chunk_score

            futures = [
                executor.submit(chunk_func, stories_list[i * K : (i + 1) * K], i) for i in range(0, n_procs)
            ]

            chunk_scores = [[] for _ in range(n_procs + 2)]
            pbars = [0] * n_procs

            for future in as_completed(futures):
                chunk, idx, pbar = future.result()
                pbars[idx] = pbar
                chunk_scores[idx] = chunk

            # pbars.reverse()

        for pbar in pbars:
            pbar.close()

        scores = [score for chunk in chunk_scores for score in chunk]
    else:
        scores, _, pbar = process_stories_chunk(stories=stories["story"].tolist(), func=func, function_id=0)
        pbar.close()

    print("", flush=True)

    return np.array(scores)


def print_mistakes(stories: pd.DataFrame, feature: str, n_shots: int, model: str):
    """Gets model outputs from cache, prints out mistakes."""
    shots = make_shots(stories, n_shots=n_shots, from_cache=True)
    prompt = FEW_SHOT_PROMPT.format(shots=shots)
    question = FEATURE_TO_PROMPT_QUESTION[feature]

    mistakes = []
    for i, row in stories.iterrows():
        input = prompt.format(story=row["story"].strip(), question=question)
        cache_id = generate_cache_id(input, model)
        cache_file_path = os.path.join(CACHE_DIR, cache_id)

        if os.path.exists(cache_file_path):
            with open(cache_file_path, "r") as cache_file:
                score, *response = cache_file.read().split("\n")

        if float(score) != row["label"]:
            mistakes.append((row["story"], row["label"], score, response))

    breakpoint()
    print("Mistakes:")
    for mistake in mistakes:
        print(mistake)


def set_all_seeds(seed):
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # Test out openai accuracy by generating small dataset, passing training
    # set through api and recording accuracy
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cache_dir", type=str, default="data", help="Adjust data cache dir")
    parser.add_argument("--feature", type=str, default="twist", help="What feature to use")
    parser.add_argument("--dataset_size", type=int, default=100, help="Size of dataset to use")
    parser.add_argument("--dataset_start_index", type=int, default=0, help="Start index for dataset")
    parser.add_argument("--few_shot", type=int, default=0, help="Number of few shot examples to use")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Which openai model to use")
    parser.add_argument("--parallelize", action="store_true", help="Whether to parallelize")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator")
    args = parser.parse_args()

    set_all_seeds(args.seed)

    if args.data_cache_dir:
        # make it a python path
        tinystories.DATA_CACHE_DIR = Path(args.data_cache_dir)

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    train, test = generate_story_classification_dataset(
        args.feature, dataset_size=args.dataset_size, dataset_start_index=args.dataset_start_index
    )
    scores = get_score_for_stories(
        stories=train,
        feature=args.feature,
        few_shot=args.few_shot,
        parallelize=args.parallelize,
        model=args.model,
    )
    # print_mistakes(train, args.feature, args.few_shot, args.model)

    # get accuracy of model by comparing score to label
    print("Accuracy of model:", flush=True)
    print(sum(scores == train["label"]) / len(scores))
    print("False positive and false negative rates:")
    print(
        sum((scores == 1.0) & (train["label"] == 0.0)) / (sum(train["label"] == 0.0) + 1),
        sum((scores == 0.0) & (train["label"] == 1.0)) / (sum(train["label"] == 1.0) + 1),
    )
    print()
    print("Incorrect elements: ", train[scores != train["label"]].index.tolist())
    # print other statistics
    print("\nOther statistics of training data:")
    print(train["label"].describe())
    print("Other statistics of scores:")
    print(pd.Series(scores).describe())
