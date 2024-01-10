import requests
import mwparserfromhell
import json
import os
from transformers import LlamaForCausalLM, LlamaTokenizerFast, AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM
import sys
import torch
from tqdm import tqdm
import traceback
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import datasets
import numpy as np
import time
import openai
from doc_info import verbalise_docs

WIKI_API_ENDPOINT = "https://en.wikipedia.org/w/api.php"

def self_info(text, model, tokenizer, merge = False):
    def merge_sub_tokens(log_probs, word_ids):
        # merge log probs of sub_tokens
        merged_log_probs = []
        current_word_id = None
        current_word_log_prob = None
        counter = 1

        for log_prob, word_id in zip(log_probs, word_ids):
            if word_id is not None:
                if current_word_id != word_id:
                    if current_word_id is not None:
                        merged_log_probs.extend([current_word_log_prob] * counter)
                    counter = 1
                    current_word_id = word_id
                    current_word_log_prob = log_prob
                else:
                    counter += 1
                    current_word_log_prob = current_word_log_prob + log_prob

        if current_word_id is not None:
            merged_log_probs.extend([current_word_log_prob] * counter)

        return merged_log_probs

    # this function is used to get the self-information of a text
    # the model should be a causal language model, e.g. GPT2LMHeadModel

    # tokenize the text
    text = f"{tokenizer.bos_token}{text}"
    encoding = tokenizer(text, return_tensors="pt", max_length=model.config.max_position_embeddings, truncation=True)
    encoding = encoding.to(model.device)

    # get the logits
    with torch.no_grad():
        logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=-1)
        info = -torch.log(probs)

    input_ids = encoding['input_ids']
    input_ids_expaned = input_ids[:, 1:].unsqueeze(-1)
    info = info[:, :-1].gather(-1, input_ids_expaned).squeeze(-1).squeeze(0).tolist()

    tokens = [tokenizer.decode(token_) for token_ in input_ids.squeeze().tolist()[1:]]
    if merge:
        info = merge_sub_tokens(info, encoding.word_ids()[1:])
    return tokens, info

def gpt3_self_info(text, num_retry = 5):
    # text = text[:1000]
    openai.api_key = os.environ["OPENAI_API_KEY"]

    for _ in range(num_retry):
        try:
            r = openai.Completion.create(
                model="curie",
                prompt=f"<|endoftext|>{text}",
                max_tokens=0,
                temperature=0,
                echo=True,
                logprobs=0,
            )
            break
        except Exception as e:
            print(e)
            time.sleep(1)

    result = r['choices'][0]
    tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

    assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

    self_info = [ -logprob for logprob in logprobs]
    # TODO: deal with the first delimiter
    return tokens, self_info

def fetch_recent_changes(from_date, to_date = '2023-08-01T00:00:00'):
    params = {
        "action": "query",
        "format": "json",
        "list": "recentchanges",
        "rcstart": to_date,  # starting from the newer date
        "rcend": from_date,  # ending at the older date
        "rctype": "new",
        "rcnamespace": "0",
        "rclimit": "500",
        "rcprop": "title|timestamp"
    }
    req = requests.Request('GET', WIKI_API_ENDPOINT, params=params).prepare()
    response = requests.get(WIKI_API_ENDPOINT, params=params).json()
    
    # Check if the response contains the expected data
    if 'query' in response and 'recentchanges' in response['query']:
        return [entry['title'] for entry in response['query']['recentchanges']]
    else:
        return []

def fetch_content(title, date=None):
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "rvlimit": "1",
    }
    if date: params["rvstart"] = date
    try:
        response = requests.get(WIKI_API_ENDPOINT, params=params)
        response.raise_for_status()  # Will raise an error if the HTTP request returned an unsuccessful status code
        data = response.json()
        if 'error' in data:
            print(f"Error fetching content for {title}: {data['error']['info']}")
            return None

        page = next(iter(data['query']['pages'].values()))
        if 'revisions' not in page:
            print(f"No revisions found for {title}")
            return None
        content = page['revisions'][0]['*']
        
        # Check if the content is a redirect and skip if true
        if content.startswith("#REDIRECT"):
            print(f"{title} is a redirect page.")
            return None
        return content

    except Exception as e:
        print(f"An error occurred while fetching content for {title}: {str(e)}")
        traceback.print_exc()  # This will print the full traceback

    return None

def parse_to_plain_text(wikitext):
    parsed = mwparserfromhell.parse(wikitext)
    return parsed.strip_code()

def select_token_window(text, token_count=400):
    tokens = text.split()
    if len(tokens) <= token_count:
        return text
    ramdom_start = np.random.randint(0, len(tokens) - token_count)
    tokens = tokens[ramdom_start:ramdom_start + token_count]
    return ' '.join(tokens)

def fetch_latest_and_historical_wiki_pages(cache_dir = '', historical_date = '2022-07-01T00:00:00Z', token_count = 300):
    # 1. Fetch the latest created pages from July 2023 and their content.
    recent_wiki_path = os.path.join(cache_dir, 'recent_wiki_pages.json')
    if not os.path.exists(recent_wiki_path):
        recent_titles = fetch_recent_changes("2023-07-01T00:00:00Z")
        recent_contents = [fetch_content(title) for title in tqdm(recent_titles)]
        recent_contents = [content for content in recent_contents if content is not None]

        data_to_save = {title: content for title, content in zip(recent_titles, recent_contents)}
        with open(recent_wiki_path, 'w') as file:
            json.dump(data_to_save, file, ensure_ascii=False, indent=4)
    else:
        with open(recent_wiki_path) as file:
            data_to_save = json.load(file)
        recent_titles = list(data_to_save.keys())
        recent_contents = list(data_to_save.values())
        recent_contents = [content for content in recent_contents if content is not None]

    # 2. Fetch a historical version of a specific title from July 2022.
    historical_wiki_path = os.path.join(cache_dir, 'historical_wiki_pages.json')
    if not os.path.exists(historical_wiki_path):
        with open(os.path.join(cache_dir, 'data/squad_wiki_title.text')) as f:
            titles = [line.strip() for line in f.readlines()]
        historical_contents = [fetch_content(title, historical_date) for title in tqdm(titles)]
        historical_contents = [content for content in historical_contents if content is not None]
        historical_to_save = {title: content for title, content in zip(titles, historical_contents)}
        with open(historical_wiki_path, 'w') as file:
            json.dump(historical_to_save, file, ensure_ascii=False, indent=4)
    else:
        with open(historical_wiki_path) as file:
            historical_to_save = json.load(file)
        historical_titles = list(historical_to_save.keys())
        historical_contents = list(historical_to_save.values())
        historical_contents = [content for content in historical_contents if content is not None]

    # 3. Parse the content to plain text.
    recent_plain_text_path = os.path.join(cache_dir, 'recent_plain_text.json')
    historical_plain_text_path = os.path.join(cache_dir, 'historical_plain_text.json')
    if not os.path.exists(recent_plain_text_path):
        plain_texts_recent = [parse_to_plain_text(content) for content in recent_contents]
        plain_texts_historical = [parse_to_plain_text(content) for content in historical_contents]
        with open(recent_plain_text_path, 'w') as file:
            json.dump(plain_texts_recent, file, ensure_ascii=False, indent=4)
        with open(historical_plain_text_path, 'w') as file:
            json.dump(plain_texts_historical, file, ensure_ascii=False, indent=4)
    else:
        with open(recent_plain_text_path) as file:
            plain_texts_recent = json.load(file)
        with open(historical_plain_text_path) as file:
            plain_texts_historical = json.load(file)

    # 4. Select a 1000-token window from the text.
    selected_windows_recent = [select_token_window(text, token_count=token_count) for text in plain_texts_recent]
    selected_windows_historical = [select_token_window(text, token_count=token_count) for text in plain_texts_historical]

    return selected_windows_recent, selected_windows_historical

def prepare_comparing_data(datasets_and_texts_col, num_samples=200, token_count=300):
    # datasets_and_texts is a dict of list {dataset_name: col_name}

    datasets_and_texts = {}
    for dataset_name, col_name in datasets_and_texts_col.items():
        if dataset_name in ['quac', 'squad_v2', 'boolq', 'iohadrubin/mini_xsum', 'liyucheng/trivia_qa_wiki_val']:
            ds = datasets.load_dataset(dataset_name, split='validation')
        elif 'RealTimeData' in dataset_name:
            ds = datasets.load_dataset(dataset_name, split='train')
        ds = ds[col_name][:num_samples]

        datasets_and_texts[dataset_name + f'_{token_count}_words'] = [select_token_window(text, token_count=token_count) for text in ds]
        # datasets_and_texts[dataset_name + '_200_words'] = [select_token_window(text, token_count=200) for text in ds]
    
    return datasets_and_texts

if __name__ == "__main__":
    cwd, model_name, token_count, = sys.argv[1:]
    token_count = int(token_count)
    batch_size = 8

    recent_snippets, historical_snippets = fetch_latest_and_historical_wiki_pages(cache_dir=cwd, token_count=token_count)
    recent_snippets = recent_snippets[:120]
    historical_snippets = historical_snippets[:120]
    wikipedia_and_texts = {
        'wiki_recent': recent_snippets,
        'wiki_historical': historical_snippets
    }
    # datasets_and_texts = prepare_comparing_data({
        # 'liyucheng/trivia_qa_wiki_val': 'wiki_context_sample'
        # 'RealTimeData/bbc_latest': 'content',
        # 'RealTimeData/bbc_2017': 'content',
        # 'iohadrubin/mini_xsum': 'document'
        # 'quac': 'context',
        # 'boolq': 'passage',
        # 'squad_v2': 'context',
        # 'RealTimeData/github_july_week1_2023': 'readme',
        # 'RealTimeData/arxiv_july_week1_2023': 'text',
        # 'RealTimeData/bbc_news_week1_july_2023': 'content',
    # }, token_count=token_count, num_samples=120)
    datasets_and_texts = verbalise_docs(num_words=token_count)

    if 'GPTQ' in model_name:
        # only llama-30b use gptq
        model = AutoGPTQForCausalLM.from_quantized(model_name, device = 'cuda:0', use_safetensors = True, disable_exllama=True if '30b' in model_name else False)
        tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
    elif 'llama' in model_name.lower():
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
        tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
    elif 'opt' in model_name.lower():
        model = OPTForCausalLM.from_pretrained(model_name, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif 'gpt2' == model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # datasets_and_texts = prepare_comparing_data({
        #     'RealTimeData/News_Seq_2021': 'maintext',
        #     'RealTimeData/News_August_2023': 'maintext',
        # })
    
    # datasets_and_texts.update(wikipedia_and_texts)

    print('=====================')
    print(f'Model: {model_name}')

    for dataset_name, texts in datasets_and_texts.items():
        print(f'=====================')
        print(f'Dataset: {dataset_name}')
        infos = []
        for text in tqdm(texts):
            try:
                if 'curie' in model_name.lower():
                    tokens, info = gpt3_self_info(text)
                else:
                    tokens, info = self_info(text, model, tokenizer)
            except:
                traceback.print_exc()
                time.sleep(10)
                continue
            # print('text:', text, '\ninfo:', info)
            infos.append(sum(info)/len(info))
        print(f'Average self-info: {sum(infos)/len(infos)}')