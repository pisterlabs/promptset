import argparse
import json
import os
import pickle
import time
from enum import Enum
from typing import Optional
from utils import Entity, Dataset, update_json_file, create_indices
import numpy as np
import openai
from tqdm import tqdm

openai.api_key = os.getenv('openai_api_key')


class ModelType(str, Enum):
    DAVINCI = "text-similarity-davinci-001"
    ADA = "text-embedding-ada-002"

    def __str__(self):
        return self.value


def resolve_model_dim_size(engine: ModelType):
    dim_size = {
        ModelType.ADA: 1536,
        ModelType.DAVINCI: 12288,
    }[engine]
    return dim_size


def try_request(model_name: str, samples: list, retries: int = 0):
    if retries >= 5:
        return
    try:
        response = openai.Embedding.create(input=samples, model=model_name)['data']
        return response
    except openai.error.InvalidRequestError:
        return
    except Exception as e:
        print(f'Retrying after catching an exception, try {retries + 1}\n{e}')
        time.sleep(5)
        return try_request(model_name, samples, retries + 1)


def prepare_dataset(dataset_name, entity, indices_path):
    input_filename = f'./data/{dataset_name}/{entity}/completions.json' if entity else f'./data/{dataset_name}/dataset.json'
    indices = create_indices(dataset_name, indices_path)

    calculated_embeddings_indices_path = f'./data/{dataset_name}/{entity}/filenames.json' if entity else f'./data/{dataset_name}/filenames.json'
    if os.path.isfile(calculated_embeddings_indices_path):
        with open(calculated_embeddings_indices_path, 'r') as f:
            finished_indices = json.load(f)
            indices = list(set(indices) ^ set(finished_indices))

    try:
        with open(input_filename, 'r') as fp:
            dataset = {k: v if isinstance(v, list) else [v] for k, v in json.load(fp).items() if k in indices}
    except KeyError:
        print(f"The file {input_filename} doesn't contain necessary keys.")
        return {}
    return dataset


def update_bin_file(filename, new_data):
    if os.path.exists(filename):
        with open(filename, "rb") as bin_file:
            existing_data = pickle.load(bin_file)
            updated_data = np.concatenate((existing_data, new_data))
    else:
        updated_data = new_data
    with open(filename, "wb") as bin_file:
        pickle.dump(updated_data, bin_file)


def write_to_file(results_folder, embeddings, successful_records, failed_requests):
    update_bin_file(f'{results_folder}/embeddings.bin', embeddings)
    update_json_file(f'{results_folder}/filenames.json', successful_records)
    update_json_file(f'{results_folder}/failed_requests.json', failed_requests)


def get_embeddings(dataset_name: Dataset,
                   engine: ModelType,
                   entity: Optional[Entity],
                   indices_path: Optional[str]
                   ):
    results_folder = f'./data/{dataset_name}/{entity}/' if entity else f'./data/{dataset_name}/'
    os.makedirs(results_folder, exist_ok=True)
    successful_records = []
    dataset = prepare_dataset(dataset_name, entity, indices_path)
    records_names = list(dataset.keys())
    num_records = len(records_names)
    if num_records == 0:
        return
    num_samples = len(dataset[list(dataset.keys())[0]]) if num_records > 1 else 1
    model_dims = resolve_model_dim_size(engine)
    embeddings = np.zeros((num_records, num_samples, model_dims), dtype=np.float32)
    with tqdm(total=num_records) as pbar:
        for i, key in enumerate(dataset):
            samples = [texti.replace("\n", " ") if isinstance(texti, str)
                       else ", ".join(texti).replace("\n", " ") if isinstance(texti, list)
                       else "none"
                       for texti in dataset[key]]
            response = try_request(engine, samples)
            if response is None:
                print(
                    f'Failed request {i} ({records_names[i]}), '
                    f'{len(dataset[key])} chars, {len(dataset[key].split())} words')
                pbar.update()
                continue
            embeddings[i] = [response[i]['embedding'] for i in range(len(response))]
            successful_records.append(records_names[i])
            pbar.update()
    del dataset
    failed_requests = list(set(records_names) - set(successful_records))
    drop_rows_mask = np.array([x not in failed_requests for x in records_names])
    embeddings = embeddings[drop_rows_mask]

    write_to_file(results_folder, embeddings, successful_records, failed_requests)


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for given messages.")

    parser.add_argument("-d", "--dataset", required=True, help="Dataset name", type=Dataset, choices=list(Dataset))
    parser.add_argument("-e", "--engine", required=True, help="OpenAI model for calculating embeddings",
                        type=ModelType, choices=list(ModelType))
    parser.add_argument("-en", "--entity_name", help="Calculate embeddings for the entity_name",
                        type=Entity, choices=list(Entity))
    parser.add_argument("-i", "--indices_path", type=str, help="Path to file with indices", default=None)

    args = parser.parse_args()

    get_embeddings(args.dataset, args.engine, args.entity_name, args.indices_path)


if __name__ == "__main__":
    main()
