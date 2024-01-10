# functions used to construct model-generated datasets based on human-written datasets from Hugging Face following https://arxiv.org/pdf/2301.11305v1.pdf
import math
import random
import transformers
from datasets import load_dataset
from multiprocessing.pool import ThreadPool


class model_text_generator:
    def __init__(self, dataset_name, base_model, base_tokenizer, cache_dir, batch_size=50, text_maximum_lenth=200, text_minimum_lenth=55, prompt_tokens=30, 
                 device='cuda', openai_model=None, openai_key=None, do_top_k=False, top_k=40, do_top_p=False, top_p=0.96):
        self.dataset_name = dataset_name
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.text_maximum_lenth = text_maximum_lenth
        self.text_minimum_lenth = 30 if dataset_name in ['pubmed'] else text_minimum_lenth
        self.prompt_tokens = prompt_tokens
        self.device = device
        self.openai_model = openai_model
        self.openai_key = openai_key
        self.do_top_k = do_top_k
        self.top_k = top_k
        self.do_top_p = do_top_p
        self.top_p = top_p

    def move_base_model_to_gpu(self):
        if self.openai_model is None:
            self.base_model.to(self.device)
        print(f'TEXT GENERATION model has moved to GPU', flush=True)


def load_language(language, cache_dir):
    # load either the english or german portion of the wmt16 dataset
    assert language in ['en', 'de']
    d = load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
    docs = d['translation']
    desired_language_docs = [d[language] for d in docs]
    lens = [len(d.split()) for d in desired_language_docs]
    sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
    return sub


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def truncate_to_substring(text, substring, idx_occurrence):
    # truncate everything after the idx_occurrence occurrence of substring
    assert idx_occurrence > 0, 'idx_occurrence must be > 0'
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]


# trim to shorter length
def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb


def _openai_sample(p, text_generator):
    import openai
    openai.api_key = text_generator.openai_key

    if text_generator.dataset_name != 'pubmed':  # keep Answer: prefix for pubmed
        p = drop_last_word(p)

    # sample from the openai model
    kwargs = { "engine": text_generator.openai_model, "max_tokens": 200}
    if text_generator.do_top_p:
        kwargs['top_p'] = text_generator.top_p
    
    r = openai.Completion.create(prompt=f"{p}", **kwargs)
    return p + r['choices'][0].text


# sample from base_model using ****only**** the first 30 tokens in each example as context
def sample_from_model(texts, text_generator, separator):
    # encode each text as a list of token ids
    if text_generator.dataset_name == 'pubmed':
        texts = [t[:t.index(separator)] for t in texts]
        all_encoded = text_generator.base_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(text_generator.device)
    else:
        all_encoded = text_generator.base_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(text_generator.device)
        all_encoded = {key: value[:, :text_generator.prompt_tokens] for key, value in all_encoded.items()}

    if text_generator.openai_model:
        # decode the prefixes back into text
        prefixes = text_generator.base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
        pool = ThreadPool(text_generator.batch_size)

        #decoded = pool.map(_openai_sample, dataset, model_collector, prefixes)
        func = functools.partial(_openai_sample, text_generator)
        decoded = pool.map(func, prefixes)
    else:
        decoded = ['' for _ in range(len(texts))]

        # sample from the model until we get a sample with at least text_minimum_lenth words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        while (m := min(len(x.split()) for x in decoded)) < text_generator.text_minimum_lenth:
            if tries != 0:
                print(f'min words: {m}, needed {text_generator.text_minimum_lenth}, regenerating (try {tries})', flush=True)

            sampling_kwargs = {}
            if text_generator.do_top_p:
                sampling_kwargs['top_p'] = text_generator.top_p
            elif text_generator.do_top_k:
                sampling_kwargs['top_k'] = text_generator.top_k
            min_length = 50 if text_generator.dataset_name in ['pubmed'] else 150
            outputs = text_generator.base_model.generate(**all_encoded, min_length=min_length, max_length=text_generator.text_maximum_lenth, do_sample=True, **sampling_kwargs, pad_token_id=text_generator.base_tokenizer.eos_token_id, eos_token_id=text_generator.base_tokenizer.eos_token_id)
            decoded = text_generator.base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            tries += 1

    return decoded


def generate_model_written_dataset(raw_data, text_generator):
    original_text = []
    sampled_text = []      

    for batch in range(math.ceil(len(raw_data) / text_generator.batch_size)): 
        print(f'Generating samples for batch {batch} of {math.ceil(len(raw_data) / text_generator.batch_size)}', flush=True)
        try:
            original_text = raw_data[batch * text_generator.batch_size:(batch + 1) * text_generator.batch_size]
        except:
            original_text = raw_data[batch * text_generator.batch_size:len(raw_data)]

        sampled_text = sample_from_model(original_text, text_generator, '<<<SEP>>>')

        for o, s in zip(original_text, sampled_text):
            if text_generator.dataset_name == 'pubmed':
                s = truncate_to_substring(s, 'Question:', 2)
                o = o.replace('<<<SEP>>>', ' ')

            o, s = trim_to_shorter_length(o, s)

            # add to the final data
            original_text.append(o)
            sampled_text.append(s)

    return original_text, sampled_text


def load_base_model_and_tokenizer(base_model_name, openai_model, dataset_name, cache_dir):
    if openai_model is None:
        print(f'Loading BASE model {base_model_name}...')
        base_model_kwargs = {}
        if 'gpt-j' in base_model_name or 'neox' in base_model_name:
            base_model_kwargs.update(dict(torch_dtype=torch.float16))
        if 'gpt-j' in base_model_name:
            base_model_kwargs.update(dict(revision='float16'))
        base_model = transformers.AutoModelForCausalLM.from_pretrained(base_model_name, **base_model_kwargs, cache_dir=cache_dir)
    else:
        base_model = None

    optional_tok_kwargs = {}
    if "facebook/opt-" in base_model_name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if dataset_name in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name, **optional_tok_kwargs, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer


def obtain_predefined_dataset(human_dataset, cache_dir):
    if human_dataset == 'xsum':
        data = load_dataset('xsum', split='train', cache_dir=cache_dir)['document']
    elif human_dataset == 'squad':
        data = load_dataset('squad', split='train', cache_dir=cache_dir)['context']
    elif human_dataset == 'pubmed':
        data = load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)
        # combine question and long_answer
        data = [f'Question: {q} Answer:<<<SEP>>>{a}' for q, a in zip(data['question'], data['long_answer'])]
    elif human_dataset == 'wmt16_en':
        data = load_language('en', cache_dir)
    elif human_dataset == 'wmt16_de':
        data = load_language('de', cache_dir)
    elif human_dataset == 'writing':
        writing_path = cache_dir + '/writingPrompts'
        if os.path.isdir(writing_path):
            with open(f'{writing_path}/valid.wp_source', 'r') as f:
                prompts = f.readlines()
            with open(f'{writing_path}/valid.wp_target', 'r') as f:
                stories = f.readlines()
            
            prompts = [process_prompt(prompt) for prompt in prompts]
            joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
            data = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]
        else:
            raise ValueError(f"Dataset writing is not existed. Please download it first and save it into './cache_dir/writingPrompts' folder")
    else:
        raise ValueError(f'Dataset {human_dataset} is not included.')

    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()
    # strip whitespace around each example
    data = [x.strip() for x in data]
    # remove newlines from each example
    data = [' '.join(x.split()) for x in data]

    random.shuffle(data)

    # try to keep only examples with > 250 words
    if human_dataset in ['xsum', 'squad', 'writing']:
        long_data = [x for x in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data

    return data


# construct model-generated datasets based on human-written datasets
def construct_predefined_dataset(human_dataset, cache_dir, generation_model_name, num_samples, openai_model):
    data = obtain_predefined_dataset(human_dataset, cache_dir)
    data = data[:num_samples]

    if openai_model is None:
        base_model_name = base_model_name.replace('/', '_')
    else:
        base_model_name = "openai-" + openai_model.replace('/', '_')

    # define generative model which generates texts
    base_model, base_tokenizer = load_base_model_and_tokenizer(base_model_name, openai_model, human_dataset, args.cache_dir)

    text_generator = model_text_generator(dataset_name=human_dataset, base_model=base_model, base_tokenizer=base_tokenizer, cache_dir=cache_dir)
    text_generator.move_base_model_to_gpu()
    original_text, sampled_text = generate_model_written_dataset(data, text_generator)
    del text_generator
    torch.cuda.empty_cache()

    return original_text, sampled_text

