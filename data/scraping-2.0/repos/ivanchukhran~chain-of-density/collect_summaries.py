import json
import os.path
from pathlib import Path

import numpy as np
import openai
from tqdm import tqdm

import datasets
from completion_request import CompletionRequest
from datasets import load_dataset, DatasetDict
from message_template import cod_message
from messages import SystemMessage, UserMessage


def gpt_summarization(config, text: str):
    response = (CompletionRequest(config)
                .add(SystemMessage(content=cod_message(5, 50)))
                .add(UserMessage(content=f"Here is the input text for you to summarise using the "
                                         f"'Missing_Entities' and 'Denser_Summary' approach:\n\n{text}"))
                .apply())
    parsed_message = json.loads(response["choices"][0]["message"]['content'])
    return parsed_message

def summarize_sample(config: dict, sample: dict):
    article = sample['article']
    summary = sample['highlights']
    gpt_summary = gpt_summarization(config, article)
    return {
        'article': article,
        'summary': summary,
        'gpt_summary': gpt_summary
    }

def collect_summaries(config: dict, dataset: DatasetDict, indices: list[int] = None):
    if indices is not None:
        dataset = [dataset[i] for i in indices]
    summaries = []
    for i in tqdm(range(len(dataset))):
        summaries.append(summarize_sample(config, dataset[i]))
    return summaries



def main():
    with open('config.json', 'r') as file:
        config = json.load(file)
    openai.api_key = config['OPENAI_API_KEY']
    DATASET_PATH = config['DATASET_PATH']
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    datasets.config.DOWNLOADED_DATASETS_PATH = Path(DATASET_PATH)
    completion_config = config['COMPLETION_CONFIG']
    dataset = load_dataset('cnn_dailymail', '3.0.0', cache_dir=DATASET_PATH)
    chosen_indices = np.random.choice(len(dataset['train']), config['NUM_SAMPLES'], replace=False).tolist()
    print(chosen_indices)
    print(len(dataset['train']))
    summaries = collect_summaries(completion_config, dataset['train'], chosen_indices)
    with open('summaries.json', 'w') as file:
        json.dump(summaries, file)


if __name__ == '__main__':
    main()

