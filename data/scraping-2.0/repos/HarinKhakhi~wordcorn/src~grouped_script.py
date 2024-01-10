import sys
import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from copy import deepcopy
import threading
from csv import DictReader

from openai import OpenAI
import utils as utils

###################### Configuration ###################### 
TOTAL_THREADS = 8

wordlist_file = sys.argv[1]
output_dir = sys.argv[2]
output_file = sys.argv[3]
operation_mode = sys.argv[4]
if not os.path.isdir(output_dir): os.makedirs(output_dir)

load_dotenv()

openai_client = OpenAI()

current_config = utils.load_configuration(file='./assets/default_config.json')
current_config = current_config.model_dump(exclude_none=True)
logger = utils.get_logger('grouped_script')

logger.info('script started...')
logger.info('current configuration: %s', current_config)
###########################################################

############################ functions ############################ 
def get_wordlist(input_file):
    wordlist = {}
    with open(input_file, 'r') as wordlist_file:
        reader = DictReader(wordlist_file)
        for row in reader:
            for group_name, word in row.items():
                if not group_name in wordlist:
                    wordlist[group_name] = []
                wordlist[group_name].append(word)
    return wordlist


def perform_task(group_name, words):
    global output_dir, operation_mode, current_config, logger, openai_client

    # check if already requested
    if (operation_mode != 'override') and os.path.isfile(f'{output_dir}/{group_name}.txt'):
        return

    # setting up configuration
    new_config = deepcopy(current_config)
    new_config['messages'].append({
        'role': 'user',
        'content': f'the list of word is {list(words)}'
    })

    # calling api
    logger.debug('calling chatgpt-api with args: %s', json.dumps(new_config, indent=4))    
    response = openai_client.chat.completions.create(
        **new_config
    )
    logger.info('got response: %s', response) 

    # writing to file
    json_object = response.choices[0].message.content
    object_file = open(f'{output_dir}/{group_name}.txt', 'w')
    object_file.write(json_object)
    object_file.close()


def combine_results(input_dir, output_file):
    def get_data(file):
        try:
            with open(file, 'r') as f: 
                return json.load(f)
        except:
            print('not json:', file)
            return {}

    arr = []
    for filename in os.listdir(input_dir):
        obj = {
            'group': filename.split('.')[0],
            **get_data(os.path.join(input_dir, filename))
        }
        arr.append(obj)

    with open(output_file, 'w') as output:
        json.dump(arr, output, indent=4)
###########################################################

grouped_wordlist = get_wordlist(wordlist_file)
total_count = len(grouped_wordlist) if operation_mode != 'test' else TOTAL_THREADS

for start_i in tqdm(range(0, total_count, TOTAL_THREADS)):
    threads = []
    for group_name in list(grouped_wordlist.keys())[start_i: start_i+TOTAL_THREADS]:
        thread = threading.Thread(target=perform_task, args=(group_name, grouped_wordlist[group_name]))
        thread.start()
        threads.append(thread)

    for thread in threads: thread.join()

combine_results(output_dir, output_file)