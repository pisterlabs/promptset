import os.path
import openai
import json
import argparse
import random
from nltk import word_tokenize
from tqdm import tqdm
import time
from retrying import retry
# from utils.tools import get_dataloader
from collections import defaultdict
import importlib
import os
import sys
sys.path.append("/scratch1/yfz5488/fairsumm") # for searching basic packages
def get_dataloader(loader):
    Loader = importlib.import_module('dataset_loader.{}'.format(loader)).Loader
    return Loader

random.seed(42)

# codex api
def call_openai_api(few_shot_prompt, inference_prompt_list, time_sleep=2, save_to="tmp.txt"):
    result = []
    pred_list = []

    # iterate prompts generated from different random seeds
    with open(save_to, 'a') as file: # Prediction is too slow, we append it in case of black out
        for i, test_prompt in tqdm(enumerate(inference_prompt_list), total=len(inference_prompt_list)):
            input_example = few_shot_prompt + test_prompt
            print(input_example)
            print("Number of tokens in this sample:", len(word_tokenize(input_example)))
            @retry(wait_exponential_multiplier=10000, wait_exponential_max=160001)
            #Wait 2^x * 10,000 milliseconds between each retry, up to 160 seconds, then 160 seconds afterwards
            def retry_create():
                # These parameters follow the guideline at https://platform.openai.com/examples/default-tldr-summary
                return openai.ChatCompletion.create(model=model,
                                                messages=[{"role": "user",
                                                          "content": input_example,}],
                                                temperature=temperature,
                                                max_tokens=max_token,
                                                top_p=1.0,
                                                frequency_penalty=0.0,
                                                presence_penalty=1)
            output = retry_create()
            # time.sleep(time_sleep)  # to slow down the requests
            tmp = output.choices[0].message['content'].strip()
            jsonl_result = {"prediction": tmp, 'input': input_example}
            result.append(jsonl_result)
            pred_list.append(tmp)
            file.write(json.dumps(jsonl_result, ensure_ascii=False) +'\n')
            file.flush()
            print(tmp)

    return pred_list

def prompt_spider_table(db_id, data_folder):
    prompt_content = ["# The information of tables:"]
    table_path=os.path.join(data_folder, "tables.json")
    schemas = json.load(open(table_path))
    schema = [x for x in schemas if x['db_id'] == db_id][0]
    for i, table in enumerate(schema['table_names_original']):
        all_column_names = [x[1] for x in schema["column_names_original"] if x[0] == i]
        single_table_prompt = f"# {i}. Table name is: {table}. The table columns are as follows: "
        for name in all_column_names:
            single_table_prompt += name + ', '
        prompt_content.append(single_table_prompt[:-2])

    prompt_content.append('\n')
    return prompt_content




if __name__ == '__main__':
    ############################# Parameters ####################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_key', type=str, default='YOUR_KEY')
    parser.add_argument('--dataset', type=str, default='Election',
                        help='keep the name of dataset the same as the data_loader name!')
    parser.add_argument("--subdataset", type=str,default="")
    parser.add_argument('--data_folder', type=str, default=
                        '/scratch1/yfz5488/fairsumm/preprocessing/preprocessed_datasets/election.json')
    parser.add_argument('--exp_name', type=str, default="10")
    parser.add_argument('--num_sent',type=int, default=-1)
    parser.add_argument('--max_token',type=int,default=512)
    parser.add_argument('--num_shot', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--split_input', type=str, default='')
    args = parser.parse_args()
    ##############################################################################
    # Prepare the parameters
    model, model_name = "gpt-3.5-turbo-16k", "text-davinci"
    openai.api_key = args.openai_key
    dataset = args.dataset
    data_folder = f"{args.data_folder}/" if 'json' not in args.data_folder else args.data_folder
    num_shot = args.num_shot
    subdataset = args.subdataset
    exp_name = args.exp_name
    num_sent = args.num_sent
    max_token = args.max_token
    temperature = args.temperature
    split_input = args.split_input
    add_fairness_claim = True
    if subdataset.__len__():
        save_folder = f"results/{dataset}/{subdataset}/"
    else:
        save_folder = f"results/{dataset}/"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_file = save_folder + f"predictions_{exp_name}.txt"

    ## if save file exists, we skip the completed ones
    start_idx = 0
    if os.path.exists(save_file):
        with open(save_file) as file:
            for line in file:
                start_idx += 1

    # read dataset (use my own data loader)
    data_loader = get_dataloader(dataset)()
    if 'RRS' in data_loader.name:
        data, label, values = data_loader.load(data_folder,
                                               remove_sep=True,
                                               test_only= num_shot==0)
    elif "Fewsum" in data_loader.name:
        data, label, values = data_loader.load(data_folder,
                                       data_name=subdataset,
                                       cluster=None) # Cluster is used for analysis
    elif "Claritin" in data_loader.name:
        data, label = data_loader.load(data_folder, male_ratio=0.8, shuffle=True)
    elif "Election" in data_loader.name:
        data, label, values = data_loader.load(data_folder)
    elif "SupremeCourt" in data_loader.name:
        data, label = data_loader.load(data_folder)
    elif "OxfordDebates" in data_loader.name:
        data, label = data_loader.load(data_folder)
    else:
        data, label, values = data_loader.load(data_folder)
    train_data, train_label = data['train'], label['train']

    # prepare prompt
    assert len(train_data) == len(train_label)
    prompt_samples_index = random.sample(list(range(len(train_data))), num_shot)
    prompt_data = [train_data[i] for i in prompt_samples_index]
    prompt_label = [train_label[i] for i in prompt_samples_index]
    prefix = []

    starter = "Article: "
    ender = "The summary of the article is:"
    if "Fewsum_amazon" in data_loader.name:
        starter = "Reviews about a product.  Each review is seperated by || :"
        ender = "The summary of the reviews is:"
    elif "Fewsum_yelp" in data_loader.name:
        starter = "Reviews about a business.  Each review is seperated by || :"
        ender = "The summary of the reviews is:"
    elif "Claritin" in data_loader.name:
        starter = "Reviews about Claritin, an anti-allergic drug. Each review is seperated by || :"
        fairness_claim = "\n80% of the reviews are written by male and 20% written by female. They are mixed randomly in " \
                         "the source text, please ensure the length of the male review in the summary is still 80% of the total length.\n"
        ender = "Please write a short text containing the salient information, i.e. a summary. The summary of the reviews is:"
        if add_fairness_claim:
            ender = fairness_claim + ender
    elif "Election" in data_loader.name:
        starter = "Reviews about US Presidential Election.  Each review is seperated by || :"
        ender = "The summary of the reviews is:"
    elif "SupremeCourt" in data_loader.name:
        starter = "Dialogue of the supreme court oral arguments. Each turn of the dialogue is one line:"
        ender = "The summary of the dialogue is:"
    elif "OxfordDebates" in data_loader.name:
        starter = "Debates of certain topic. Each turn of the dialogue is one line:"
        ender = "The summary of the dialogue is:"
    if num_sent != -1:
        ender = ender + " Summary it in {} sentences.".format(num_sent)

    for source, target in zip(prompt_data[start_idx:], prompt_label[start_idx:]):
        if data_loader.is_dialogue:
            source = '\n'.join(source)
        else:
            source = ' || '.join(source)
        prefix.append(f"{starter}\n{source}")
        prefix.append(f"\n {ender}")
        prefix.append(f"{target}\n")

    prefix = "\n".join(prefix) + '\n'

    # prepare test samples
    test_data, test_label, test_values = data['test'], label['test'], values['test'] # starting from the half
    test_sample = []

    for source, value in zip(test_data[start_idx:], test_values[start_idx:]):
        cur_sample = []

        if split_input == 'sentiment':
            source_value_dict = defaultdict(list)
            for s, v in zip(source, value):
                source_value_dict[v].append(s)
            source = ""
            for k, v in source_value_dict.items():
                if "Fewsum_yelp" in data_loader.name:
                    mapping = {'neg': 'negative', 'pos': 'positive', 'neu': 'neutral'}
                    source += 'These are the reviews that are {}:\n'.format(mapping[k])
                    source += " || ".join(v)
                    source += '\n'

                else:
                    print("Not implemented!!!")

        else:
            if data_loader.is_dialogue:
                source = '\n'.join(source)
            else:
                source = ' || '.join(source)

        if add_fairness_claim is True:
            if "Fewsum_yelp" in data_loader.name:
                # fairness_claim = ("please ensure the number of words of the postive/negative/neutral contents in "
                #                   "the summary is the same as the ratio that in source.\n")
                fairness_claim = ("\nNext you will do a summarization for the given reviews. You need to ensure the fairness of the generatered summary."
                                  "The ratio of positive and negative content in the summary should be the same as source.\n")
                # ratio = 'That is, you need to ensure the same as the ratio of: '
                # for k, v in source_value_dict.items():
                #     ratio += f"{mapping[k]} has {len(v)} reviews;"
                # fairness_claim += ratio
            if "Election" in data_loader.name:
                fairness_claim = (
                    "\n\nInstruction: Next you will do a summarization for the given reviews. You need to ensure the fairness of the generatered summary."
                    "The ratio of content written by democratic and republican supporters in the summary should be the same as the ratio in source.\n")
                source += fairness_claim

        cur_sample.append(f"{starter}\n{source}")
        cur_sample.append(f"\n {ender}")
        cur_sample = '\n'.join(cur_sample)
        test_sample.append(cur_sample)

    # call the API!
    results = call_openai_api(prefix, test_sample, time_sleep=3, save_to=save_file)
    new_results = []
    for result, sample in zip(results, test_data[start_idx:]):
        new_results.append(result.update(sample))
    json_save_file = save_file.replace("txt", 'json')
    with open(json_save_file, 'a') as file:
        new_results = [json.dumps(x, ensure_ascii=False) for x in new_results]
        file.write('\n'.join(new_results))

    print("Finish, total number of generated samples:", len(new_results))
