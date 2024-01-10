import os.path
import openai
import json
import argparse
import random
from tqdm import tqdm
import time
from retrying import retry
random.seed(42)

# codex api
def call_openai_api(few_shot_prompt, inference_prompt_list, time_sleep=2, save_to="tmp.txt"):
    result = []
    pred_list = []

    # iterate prompts generated from different random seeds
    with open(save_to, 'a') as file: # Prediction is too slow, we append it in case of black out
        for i, test_prompt in tqdm(enumerate(inference_prompt_list), total=len(inference_prompt_list)):
            input_example = few_shot_prompt + test_prompt
            @retry(wait_exponential_multiplier=10000, wait_exponential_max=160001)
            #Wait 2^x * 10,000 milliseconds between each retry, up to 160 seconds, then 160 seconds afterwards
            def retry_create():
                return openai.Completion.create(model=model, prompt=input_example, max_tokens=256, temperature=0)
            output = retry_create()
            time.sleep(time_sleep)  # to slow down the requests
            tmp = output.choices[0].text.split("\n")[0][1:].strip()
            result.append({"prediction": tmp})
            pred_list.append(tmp)
            file.write(tmp+'\n')
            file.flush()
            print(f"{language}_{mr} ", tmp)

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
    parser.add_argument('--openai_key', type=str, default='')
    parser.add_argument('--dataset', type=str, default='mgeoquery')
    parser.add_argument('--data_folder', type=str, default='../../dataset/')
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--mr', type=str, default='sql')
    parser.add_argument('--num_shot', type=int, default=8)
    parser.add_argument('--crosslingual', action='store_true', default=False)

    args = parser.parse_args()
    ##############################################################################
    # Prepare the parameters
    model, model_name = "code-davinci-002", "codex_davinci"
    openai.api_key = args.openai_key
    dataset = args.dataset
    data_folder = f"{args.data_folder}{dataset}/"
    language = args.language
    mr = args.mr
    num_shot = args.num_shot
    cross_lingual = args.crosslingual

    if dataset == 'mcwq':
        data_folder = data_folder + "mcd3/"

    # prepare write file
    if cross_lingual:
        save_folder = f"results/crosslingual/{dataset}/"
    else:
        save_folder = f"results/monolingual/{dataset}/" # TODO: add a monolingual folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_file = save_folder + f"{language}_{mr}.txt"
    ## if save file exists, we do the summarization
    start_idx = 0
    if os.path.exists(save_file):
        with open(save_file) as file:
            for line in file:
                start_idx += 1

    # prepare prompt
    train_data = json.load(open(data_folder+"train.json"))
    prompt_samples = random.sample(train_data, num_shot)
    prefix = []
    ## Special case for crosslingual setting
    if cross_lingual:
        print("Crosslingual Setting, use english as prompt")
        prompt_lang = 'en'
    else:
        prompt_lang = language
    for sample in prompt_samples:
        prefix.append(f"# Translate the following sentences into {mr}:\n\n")
        prefix.append("# Question:")
        prefix.append(f"# {sample['question'][prompt_lang]}\n")
        # Spider dataset has dataset to append
        if dataset == 'mspider':
            prefix += prompt_spider_table(sample['db_id'], data_folder)
        prefix.append("# Translation results are as follows:")
        if isinstance(sample['mr'][mr], str):
            prefix.append(f"# {sample['mr'][mr]}\n\n")
            # mgeoquery
        else:
            prefix.append(f"# {sample['mr'][mr][prompt_lang]}\n\n")
            # mtop
    prefix = "\n".join(prefix)

    # prepare test samples
    test_path = data_folder+"test.json"
    if not os.path.exists(test_path):
        test_path = data_folder+"dev.json"

    if dataset == "mconala":
        if cross_lingual:
            test_path = data_folder+"test.json" # If cross lingual, then only test set
        else:
            test_path = data_folder+'dev.json' # If monolingual we use dev with en samples

    test_data = json.load(open(test_path))[start_idx:] # starting from the half
    test_sample = []

    for sample in test_data:
        cur_sample = []
        cur_sample.append(f"# Translate the following sentences into {mr}:\n\n")
        # Spider dataset has dataset to append
        if dataset == 'mspider':
            cur_sample += prompt_spider_table(sample['db_id'], data_folder)
        cur_sample.append("# Question:")
        if dataset == 'mconala':
            avaliable_lang = list(sample['question'].keys())[0]
            cur_sample.append(f"# {sample['question'][avaliable_lang]}")
        else:
            cur_sample.append(f"# {sample['question'][language]}\n")
        cur_sample.append("# Translation results are as follows:\n")
        cur_sample = '\n'.join(cur_sample)
        test_sample.append(cur_sample)

    # call the API!
    result = call_openai_api(prefix, test_sample, time_sleep=3, save_to=save_file)
    print("Finish, total number of generated samples:", len(result))
