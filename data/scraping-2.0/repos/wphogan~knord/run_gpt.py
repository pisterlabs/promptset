import logging
import os
import random
import time

from torch.utils.data import DataLoader

from baselines.gpt import ChatApp
from utils.utils_dataset_gpt import OpenAIDataset
from utils.utils_gpt import GPTExamples, load_jsonl_data, rel_code2instances, get_processed_uids

timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

logging.basicConfig(level=logging.INFO, filename=f'logs/gpt_prompt_{timestamp}.log', filemode='w')


def main():
    # Get UIDs from instances we already processed
    dataset_name = 'retacred'
    uids_to_ignore = get_processed_uids(f'data/{dataset_name}/{dataset_name}_unlabeled_gpt_predictions.jsonl')

    # Make sure this run is good to go
    print(f'{dataset_name} -- Loaded {len(uids_to_ignore)} uids to ignore. (Already processed)')
    x = input('Continue? (y/n)')
    if x != 'y':
        print('Exiting...')
        exit()

    # Init classes
    gpt_examples = GPTExamples(dataset=dataset_name)

    # Load data and shuffle
    random.seed(42)
    data_labeled = load_jsonl_data(f'data/{dataset_name}/{dataset_name}_label_nonorel.jsonl')
    data_unlabeled = load_jsonl_data(f'data/{dataset_name}/{dataset_name}_unlabel_nonorel.jsonl')
    random.shuffle(data_labeled)
    random.shuffle(data_unlabeled)

    rel_code2instances_labeled = rel_code2instances(data_labeled)

    # Dataloader
    dataset_unlabeled = OpenAIDataset(data_unlabeled, dataset_name=dataset_name)
    dataloader_unlabeled = DataLoader(dataset_unlabeled, batch_size=1, shuffle=False)

    # Init chat app
    chat_app = ChatApp(rel_code2instances_labeled, gpt_examples, uids_to_ignore=uids_to_ignore, dataset=dataset_name)

    # Run GPT-3.5
    chat_app.prompt_send_and_receive(dataloader_unlabeled, gpt_examples)


if __name__ == '__main__':
    print('Starting file: ', os.path.basename(__file__))
    main()
    print('\nCompelted file: ', os.path.basename(__file__))
