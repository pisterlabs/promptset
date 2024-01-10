import asyncio
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import tiktoken
import time

from gpt_utils import (
    AsyncGPTChat,
    manage_chat_async,
    FIRST_PROMPT,
    SECOND_PROMPT,
)
from utils import (
    get_prompt_and_labels,
    extract_predictions,
    get_accuracy,
    load_from_jsonl,
    process_results_table,
)
from datasets import load_dataset
from dataset_utils import (
    get_dataset_numbers,
    get_dataset_capitalized,
    get_dataset_primes,
    get_dataset_even,
)


async def run_chats(
    datasets: dict,
    number_of_chats_per_dataset: int = 5,
    n_shots: int = 1,
    test_size: int = 10,
    model="gpt-3.5-turbo",
) -> dict:
    number_of_chats = len(datasets) * number_of_chats_per_dataset
    print(f"\nRunning {number_of_chats} chats in parallel...")
    chats = [
        AsyncGPTChat(system_message="", model=model) for _ in range(number_of_chats)
    ]
    prompts_list = []
    labels_list = []

    for dataset in datasets.values():
        prompt, labels = get_prompt_and_labels(
            dataset, FIRST_PROMPT, n_shots=n_shots, test_size=test_size
        )
        for _ in range(number_of_chats_per_dataset):
            prompts_list.append(prompt)
            labels_list.append(labels)

    print("Total tokens Part 1: ", count_tokens(prompts_list))
    start_time = time.time()
    results = await asyncio.gather(
        *(manage_chat_async(chat, prompt) for chat, prompt in zip(chats, prompts_list))
    )

    # Process results
    outputs = [extract_predictions(response) for response in results]

    # We are going to to use only conversations where the accuracy has crossed the threshold, otherwise we get limit problems from openai
    accuracies = [
        get_accuracy(labels, preds) for labels, preds in zip(labels_list, outputs)
    ]
    accuracy_threshold_indices = [i for i, acc in enumerate(accuracies) if acc >= 0.9]

    filtered_chats = [chats[i] for i in accuracy_threshold_indices]
    print("Len of filtered chats: ", len(filtered_chats))
    if filtered_chats:
        filtered_explanations = await asyncio.gather(
            *(manage_chat_async(chat, SECOND_PROMPT) for chat in filtered_chats)
        )
    else:
        filtered_explanations = []

    # Set all explanations below the threshold to None
    explanations = [None] * len(accuracies)
    for index, explanation in zip(accuracy_threshold_indices, filtered_explanations):
        explanations[index] = explanation

    end_time = time.time() - start_time
    print(f"Async Chats took {end_time:.2f} seconds")

    results_dict = {}
    for name, i in zip(
        datasets.keys(), range(0, len(outputs), number_of_chats_per_dataset)
    ):
        results_dict[name] = {}
        results_dict[name]["preds"] = outputs[i : i + number_of_chats_per_dataset]
        results_dict[name]["labels"] = labels_list[i : i + number_of_chats_per_dataset]
        results_dict[name]["explanations"] = explanations[
            i : i + number_of_chats_per_dataset
        ]

    # return results, results_dict, prompts_list
    return results_dict


if __name__ == "__main__":
    # # GPT-3.5 turbo setup
    # number_of_chats_per_dataset = 4
    # n_shots = [2, 4, 8, 16, 24]
    # test_size = 50
    # model = "gpt-3.5-turbo"

    # GPT-4 setup
    number_of_chats_per_dataset = 1
    n_shots = [4, 10, 24]
    test_size = 50
    model = "gpt-4-1106-preview"

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def count_tokens(l: list):
        return sum([len(encoding.encode(s)) for s in l])

    datasets = {}
    dataset = load_dataset("imdb", split="train")
    datasets["Numbers"] = get_dataset_numbers(
        dataset, size=100, true_proportion=0.5, str_length=15
    )
    datasets["Capitals"] = get_dataset_capitalized(dataset, size=100, str_length=15)
    datasets["Even_nums"] = get_dataset_even(size=1000)
    datasets["Primes"] = get_dataset_primes(3, 200)
    datasets["Fruits"] = load_from_jsonl("data/fruits_and_vegetables.jsonl")
    datasets["Contradictions"] = load_from_jsonl("data/contradictions.jsonl")


    results_dict = {}
    df_dict = []
    for n_shot in tqdm(n_shots):
        results_dict[n_shot] = asyncio.run(
            run_chats(
                datasets,
                number_of_chats_per_dataset=number_of_chats_per_dataset,
                n_shots=n_shot,
                test_size=test_size,
                model=model,
            )
        )
        df_dict.append(process_results_table(results_dict[n_shot], n_shots=n_shot))
        time.sleep(20)  # prevent hitting API limits

    df = pd.concat(df_dict, axis=1)
    columns = [f"accuracy_{i}" for i in n_shots] + [f"explanation_{i}" for i in n_shots]
    df = df[columns]
    df.to_parquet(f"results_df_{model}.parquet")
