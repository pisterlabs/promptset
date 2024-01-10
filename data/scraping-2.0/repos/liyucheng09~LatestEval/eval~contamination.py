import datasets
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import os
import time
import openai
from tqdm import tqdm
from transformers import GPT2TokenizerFast

T = GPT2TokenizerFast.from_pretrained("gpt2")
prompt_length = 250
suffix_length = 500 - prompt_length

def data_sampler():
    quac = datasets.load_dataset("quac", split="validation")
    boolq = datasets.load_dataset("boolq", split="validation")
    squad = datasets.load_dataset("squad_v2", split="validation")

    latesteval_1 = datasets.load_dataset("RealTimeData/bbc_news_week1_july_2023", split="train")
    latesteval_2 = datasets.load_dataset("RealTimeData/github_july_week1_2023", split="train")
    latesteval_3 = datasets.load_dataset("RealTimeData/arxiv_july_week1_2023", split="train")

    def get_prefix_and_suffix(doc, dataset_name = None):
        if dataset_name is None:
            raise ValueError("dataset_name must be specified")
        if dataset_name == "quac" or dataset_name == "squad_v2":
            text = T(doc['context']).input_ids
            if dataset_name == "quac":
                title = 'quac, ' + doc['wikipedia_page_title'] + ', ' + doc['section_title'] + ', '
            elif dataset_name == "squad_v2":
                title = 'squadv2, ' + 'wikipedia, ' + doc['title'] + ', '
            # text = word_tokenize(doc['context'])
        elif dataset_name == "boolq":
            text = T(doc['passage']).input_ids
            title =  'boolq, wikipedia, '
            # text = word_tokenize(doc['passage'])
        elif dataset_name == "latesteval_1":
            text = doc['content'].replace("\n", " ")
            text = T(text).input_ids
            # text = word_tokenize(doc['content'])
            title = 'bbc, '
        if len(text) > 1000:
            prefix = T.decode(text[:prompt_length])
            suffix = T.decode(text[suffix_length:])
        else:
            suffix = T.decode(text[-suffix_length:])
            prefix = T.decode(text[: -suffix_length])
        # prefix = " ".join(prefix)
        # suffix = " ".join(suffix)
        prefix = title + prefix
        return pd.Series([prefix, suffix], index=['prefix', 'suffix'])

    # quac = quac.to_pandas().sample(n=10, random_state=42)
    # boolq = boolq.to_pandas().sample(n=100, random_state=42)
    # squad = squad.to_pandas().sample(n=100, random_state=42)
    # latesteval_1 = latesteval_1.to_pandas().sample(n=100, random_state=42)

    quac = quac.to_pandas().head(n=100)
    boolq = boolq.to_pandas().head(n=100)
    squad = squad.to_pandas().head(n=100)
    latesteval_1 = latesteval_1.to_pandas().head(n=30)

    quac = quac.apply(get_prefix_and_suffix, axis=1, dataset_name="quac")
    boolq = boolq.apply(get_prefix_and_suffix, axis=1, dataset_name="boolq")
    squad = squad.apply(get_prefix_and_suffix, axis=1, dataset_name="squad_v2")
    latesteval = latesteval_1.apply(get_prefix_and_suffix, axis=1, dataset_name="latesteval_1")

    return {
        "quac": quac,
        "boolq": boolq,
        "squad": squad,
        "latesteval": latesteval
    }

def identify_contamination(reference_suffixes, continuations):
    
    def generate_word_ngrams(text, n, use_lemmatization=False):
        tokens = T(text.lower()).input_ids  
        
        # Optionally, lemmatize words
        if use_lemmatization:
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]
        return list(ngrams(tokens, n))

    results = []
    for suffix, continuation in zip(reference_suffixes, continuations):
        suffix_ngrams = set(generate_word_ngrams(suffix, 9))
        continuation_ngrams = set(generate_word_ngrams(continuation, 9))

        intersection = suffix_ngrams.intersection(continuation_ngrams)

        if len(intersection) > 0:
            results.append((True, suffix, continuation, intersection))
    
    return results

def generate_continuation(model, prompts, reference_suffix, benchmark, batch_size=10):
    # three models at this moment: gpt-3, gpt-4, llama-2

    prompts = prompts.tolist()

    if model in ['gpt-4', 'davinci', 'curie', 'babbage']:
        generate = gpt
    else:
        generate = hf_generate
    
    continuations = []
    output_file = f"eval/{model}_{benchmark}_{prompt_length}_continuation.txt"
    prompt_file = f"eval/{model}_{benchmark}_{prompt_length}_prompt.txt"
    reference_suffix_file = f"eval/{model}_{benchmark}_{prompt_length}_reference_suffix.txt"
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            continuations = f.readlines()
        return continuations
    else:
        with open(output_file, "w") as f, open(prompt_file, "w") as f2, open(reference_suffix_file, "w") as f3:
            for i in tqdm(range(0, len(prompts), batch_size)):
                prompt = prompts[i: i + batch_size]
                reference_suffix_batch = reference_suffix[i: i + batch_size]
                continuation = generate(prompt, model=model)
                continuations.extend(continuation)
                f.write('\n'.join(continuation) + "\n")
                f2.write('\n'.join(prompt) + "\n")
                f3.write('\n'.join(reference_suffix_batch) + "\n")
        
        return continuations

def hf_generate(model, prompt):
    pass

def gpt(prompt, num_retry = 5, model = "gpt-3.5-turbo"):
    # generate answer by gpt-3.5-turbo
    openai_key = os.environ.get("OPENAI_API_KEY")
    for _ in range(num_retry):
        try:
            if model in ['davinci', 'curie', 'babbage']:
                r = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=250,
                    temperature=0,
                    logit_bias={"198": -100},
                    logprobs=0,
                )
            elif model in ['gpt-3.5-turbo', 'gpt-4']:
                r = openai.ChatCompletion.create(
                    model = model,
                    messages = [
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=250,
                    temperature = 0,
                    logit_bias={"198": -100}
                )
            break
        except Exception as e:
            print(e)
            time.sleep(1)
    
    if model in ['davinci', 'curie', 'babbage']:
        return [x['text'].replace('\n', ' ') for x in r['choices']]
    elif model in ['gpt-3.5-turbo', 'gpt-4']:
        return [x['message']['content'] for x in r['choices']]

if __name__ == "__main__":
    samples = data_sampler()

    quac = samples['quac']
    boolq = samples['boolq']
    squad = samples['squad']
    latesteval = samples['latesteval']

    model = 'curie'

    quac_continuations = generate_continuation(model, quac['prefix'], quac['suffix'], "quac")
    quac_results = identify_contamination(quac['suffix'], quac_continuations)

    print(f"-- quac: {len(quac_results)}, -- {len(quac_results) / len(quac)}")

    boolq_continuations = generate_continuation(model, boolq['prefix'], boolq['suffix'], "boolq")
    boolq_results = identify_contamination(boolq['suffix'], boolq_continuations)

    print(f"-- boolq: {len(boolq_results)}, -- {len(boolq_results) / len(boolq)}")

    squad_continuations = generate_continuation(model, squad['prefix'], squad['suffix'], "squad")
    squad_results = identify_contamination(squad['suffix'], squad_continuations)

    print(f"-- squad: {len(squad_results)}, -- {len(squad_results) / len(squad)}")

    latesteval_continuations = generate_continuation(model, latesteval['prefix'], latesteval['suffix'], "latesteval")
    latesteval_results = identify_contamination(latesteval['suffix'], latesteval_continuations)

    print(f"-- latesteval: {len(latesteval_results)}, -- {len(latesteval_results) / len(latesteval)}")