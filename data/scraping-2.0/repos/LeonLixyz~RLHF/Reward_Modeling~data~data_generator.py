import json
import openai
from tqdm import trange, tqdm
import random
import datasets

openai.organization = "org-eWVexeCXCmgZwXvzGpThQFWf"
openai.api_key = "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"


def split_and_save_data(dataset_name, split_sft=1):
    data_dir_formatted = f"./data/dataset/{dataset_name}.json"
    # Load the data
    with open(data_dir_formatted, "r") as f:
        annotated = json.load(f)

    # Shuffle and split the data
    random.shuffle(annotated)
    dataset_len = len(annotated)
    eval_len = 512
    train_len = dataset_len - eval_len
    train_annotated = annotated[:train_len]
    eval_annotated = annotated[train_len:]

    if split_sft != 1:
        sft_split = int(len(train_annotated) * split_sft)
        sft_annotated = train_annotated[: sft_split]
        train_annotated = train_annotated[sft_split :]

        # process sft data, we only want the preferred output
        for data in sft_annotated:
            data['output'] = data['output_1'] if data['preference'] == 1 else data['output_2']
            data.pop('output_1')
            data.pop('output_2')
            data.pop('preference')

        # convert the sft data to huggingface datasets format
        sft_annotated_datasets = datasets.Dataset.from_list(sft_annotated)
        # rename train split name from train to sft
        

        sft_annotated_datasets.push_to_hub(f"{dataset_name}_sft")

        with open(f"./data/dataset/{dataset_name}/sft.json", "w") as f:
            json.dump(sft_annotated, f, ensure_ascii=False, indent=4)

    # Save the split data to separate JSON files
    with open(f"./data/dataset/{dataset_name}/train.json", "w") as f:
        json.dump(train_annotated, f, ensure_ascii=False, indent=4)
    with open(f"./data/dataset/{dataset_name}/eval.json", "w") as f:
        json.dump(eval_annotated, f, ensure_ascii=False, indent=4)


def process_aug_output(output):
    try:
        # we split the string into favored and less favored responses
        splitted = output.split("\n\n###")
        favored, less_favored = splitted[0], splitted[1]
        # here we split it to individual responses

        favored_responses = favored.split("[[[")
        less_favored_responses = less_favored.split("[[[")

        # here we remove the first element of the list, because it's filler title
        favored_responses = favored_responses[1:]
        less_favored_responses = less_favored_responses[1:]

        # here we remove the last few characters at the back until ]]] to remove the filler title
        def remove_end_filler(response):
            return response.split("]]]")[0]

        favored_responses = [
            remove_end_filler(response) for response in favored_responses
        ]
        less_favored_responses = [
            remove_end_filler(response) for response in less_favored_responses
        ]

        random.shuffle(favored_responses)
        random.shuffle(less_favored_responses)

        return favored_responses, less_favored_responses
    except:
        return None, None


def generate_augmented_data(dataset, number_of_copies, dataset_name):
    augmented_data = []

    for data in tqdm(dataset):
        instruction = data["instruction"]
        input = data["input"]
        output_1 = data["output_1"]
        output_2 = data["output_2"]
        preference = data["preference"]

        favored_response = output_1 if preference == 1 else output_2
        less_favored_response = output_2 if preference == 1 else output_1
        message = f"""Your objective is to rephrase sentences while maintaining the original's quality, semantics, and meaning. Do not enhance or diminish the quality of the original content.\nFor each set of instruction and input below, you will find a favored response and a less favored response. Your task is to generate {number_of_copies} alternative phrasings for each response, ensuring they preserve the same meaning and hold consistent with the original's quality, neither elevating nor diminishing it. The output format should be \n\n###Favored Responses:\n##1. [[[Response 1]]]\n##2. [[[Response 2]]]\n##3. [[[Response 3]]]\n...\n\n###Less Favored Responses:\n##1. [[[Response 1]]]\n##2. [[[Response 2]]]\n##3. [[[Response 3]]]\n... Each response should be wrapped in [[[Response]]] \n\n###Instruction:\n{instruction}\n\n###Input:\n{input}\n\n###Favored Response:\n{favored_response}\n\n###Less Favored Response:\n{less_favored_response}"""
        payload = [
            {
                "role": "system",
                "content": "You are a linguistic agent with the task of paraphrasing sentences.",
            },
            {"role": "user", "content": f"{message}"},
        ]

        retries = 3
        while retries > 0:
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-4", messages=payload, temperature=1
                )
                output = completion["choices"][0].message["content"]
                break
            except Exception as e:
                retries -= 1
                print(e)
                print(f"Retrying for {instruction}...")
                continue

        # process the output into favored and less favored responses
        favored_responses, less_favored_responses = process_aug_output(output)
        # if the output is not in the correct format, we skip it
        if favored_responses is None or less_favored_responses is None:
            continue

        output_1_augmented = (
            favored_responses if preference == 1 else less_favored_responses
        )
        output_2_augmented = (
            less_favored_responses if preference == 1 else favored_responses
        )

        augmented_data.append(
            {
                "instruction": instruction,
                "input": input,
                "output_1": output_1,
                "output_2": output_2,
                "output_1_augmented": output_1_augmented,
                "output_2_augmented": output_2_augmented,
                "preference": preference,
                "raw_augmented_output": output,
            }
        )

        with open(f"./data/dataset/{dataset_name}/augmented.json", "w") as f:
            json.dump(augmented_data, f, ensure_ascii=False, indent=4)


def generate_dyadic_batch(dataset_name, dataset, batch_length=10):
    """
    Generate batch of augmented data for input dataset.
    Where output is a list of list of dictionaries, where each list of size batch_length, are dictionaries is a batch of augmented data.
    each batch contains a random permutation of augmented data of data.
    """

    # Initialize the list to store the augmented data
    augmented_batch = []
    dataset_length = len(dataset)
    for i in trange(0, dataset_length, 2):
        dyadic_batch = []
        # random order gives list of order of dyadic pairs. eg [1,1,2,1,2,1,1,2]
        random_order = [random.randint(i, i + 1) for _ in range(batch_length)]
        for j in random_order:
            data = dataset[j]
            instruction = data["instruction"]
            input = data["input"]
            original_output_1 = data["output_1"]
            original_output_2 = data["output_2"]
            preference = data["preference"]
            augmented_output_1 = data["output_1_augmented"]
            augmented_output_2 = data["output_2_augmented"]

            # Randomly select augmented output
            try:
                sim_output_1 = random.choice(augmented_output_1)
                sim_output_2 = random.choice(augmented_output_2)
            except:
                print("data has no augmented output", data)
                continue
            # Save the generated outputs
            augmented_data_dict = {
                "instruction": instruction,
                "input": input,
                "preference": preference,
                "output_1": sim_output_1,
                "output_2": sim_output_2,
            }

            dyadic_batch.append(augmented_data_dict)
        augmented_batch.append(dyadic_batch)

        # Append augmented batch to augmented data json file

def generate_unaugmented_dyadic_batch(dataset_name, dataset, batch_length=10):
    """
    Generate batch of augmented data for input dataset.
    Where output is a list of list of dictionaries, where each list of size batch_length, are dictionaries is a batch of augmented data.
    each batch contains a random permutation of augmented data of data.
    """

    # Initialize the list to store the augmented data
    augmented_batch = []
    dataset_length = len(dataset)
    for i in trange(0, dataset_length, 2):
        dyadic_batch = []
        # random order gives list of order of dyadic pairs. eg [1,1,2,1,2,1,1,2]
        random_order = [random.randint(i, i + 1) for _ in range(batch_length)]
        for j in random_order:
            data = dataset[j]
            instruction = data["instruction"]
            input = data["input"]
            original_output_1 = data["output_1"]
            original_output_2 = data["output_2"]
            preference = data["preference"]

            # Save the generated outputs
            augmented_data_dict = {
                "instruction": instruction,
                "input": input,
                "preference": preference,
                "output_1": original_output_1,
                "output_2": original_output_2,
            }

            dyadic_batch.append(augmented_data_dict)
        augmented_batch.append(dyadic_batch)

        # Append augmented batch to augmented data json file
        with open(f"./data/dataset/{dataset_name}/unaug_joint_eval.json", "w") as f:
            json.dump(augmented_batch, f, indent=4, sort_keys=True)

def main():
    random.seed(42)
    dataset_name = "anthropic_hh"
    number_of_augmented_data = 10

    split_and_save_data(dataset_name, split_sft=0.5)

    dataset = f"./data/dataset/{dataset_name}/eval.json"

    with open(dataset, "r") as f:
        annotated = json.load(f)

    # !! Comment out below if you don't want to generate augmented data
    # generate_augmented_data(annotated, number_of_augmented_data, dataset_name)

    # with open(f"./data/dataset/{dataset_name}/augmented.json", "r") as f:
    #     augmented_dataset = json.load(f)
    # generate_dyadic_batch(dataset_name, augmented_dataset, batch_length=10)

    # !! End Comment

    generate_unaugmented_dyadic_batch(dataset_name, annotated, batch_length=10)


if __name__ == "__main__":
    main()
