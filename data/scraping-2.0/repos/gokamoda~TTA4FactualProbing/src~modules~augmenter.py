import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import itertools
from omegaconf import DictConfig
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import openai
import re
import os



from textattack.augmentation import Augmenter, WordNetAugmenter
from textattack.transformations import WordSwapEmbedding
import texthero as hero

from modules.models import get_model

tqdm.pandas()

# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=True)

class PromptDataset(Dataset):
    def __init__(self, tokenized_texts, attention_mask):
        self.tokenized_texts = tokenized_texts
        self.attention_mask = attention_mask


    def __len__(self):
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx):
        return self.tokenized_texts[idx], self.attention_mask[idx]

def split_list(sequence: list, num_cols = None, num_rows = None):
    assert sequence != None
    sequence_len = len(sequence)
    if num_cols != None and num_rows != None:
        assert num_cols * num_rows == sequence_len, "need num_cols * num_rows == sequence_len"
    if num_cols == None:
        assert num_rows != None, "at least one of num_cols or num_rows need to be set"
        assert sequence_len % num_rows == 0, "sequence length not multiple of num_rows"
        num_cols = int(sequence_len / num_rows)
        
    return [sequence[i:i+num_cols] for i in range(0, sequence_len, num_cols)]



def augment_head(original_df: pd.DataFrame, method_name: str, args: DictConfig) -> pd.DataFrame:
    augmenters = {
        'back_translation': back_translation,
        'word_swapping': word_swapping,
        'stopwords_filtering': stopword_filtering,
        'openai': openai_paraphrase
    }
    
    augmented_prompts: pd.DataFrame = augmenters[method_name](original_df.copy(), **args)
    return augmented_prompts

def openai_paraphrase(original_df: pd.DataFrame, model: str, num_return_sequences: int, label: str) -> pd.DataFrame:
     
    args = {
        'model': model,
        'max_tokens': 1024,
        'temperature': 0.8,
        'top_p': 1,
        'frequency_penalty': 0,
        'presence_penalty': 0
    }
    format_pattern = re.compile(r'^[0-9]+.(.+)')
    openai.api_key = os.environ['CHATGPT_API']

    def paraphrase(row):
        original_prompt = row['prompt']
        prompt = "Would you provide 10 paraphrases for the following question?\n{}".format(original_prompt)

        response = None
        received = False
        while not received:
            try:
                response = openai.Completion.create(
                    prompt=prompt,
                    **args
                )
                received = True
            except:
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(1)
        
        prompts = response["choices"][0]["text"]
        prompts = prompts.strip().split('\n')
        prompts = [format_pattern.match(prompt).group(1).strip() for prompt in prompts]
        n_more_prompts = num_return_sequences - len(prompts)
        prompts += ['']*n_more_prompts
        assert len(prompts) == num_return_sequences, 'could not get enough augmented prompts'
        row['augmented_prompt'] = prompts
        return row

    original_df = original_df.apply(paraphrase, axis=1)

    augmented_prompts = original_df['augmented_prompt'].to_list()
    augmented_prompts = list(itertools.chain.from_iterable(augmented_prompts))

    assert len(augmented_prompts) == num_return_sequences * original_df.shape[0], 'could not get enough augmented prompts'
    
    fact_id = [x for x in range(0, original_df.shape[0]) for _ in range(num_return_sequences)]
    augmented_prompts_df = pd.DataFrame({'fact_id': fact_id, 'score': 1, 'prompt': augmented_prompts, 'label': label})

    return augmented_prompts

def word_swapping(original_df: pd.DataFrame, type: str, num_return_sequences: int, label: str) -> pd.DataFrame:
    augmenters = {
        'wordswap': Augmenter(transformation = WordSwapEmbedding()),
        'wordnet': WordNetAugmenter(pct_words_to_swap=0.2)
    }
    augmenter = augmenters[type]
    augmenter.transformations_per_example = num_return_sequences

    def augment(row):
        augment_result = augmenter.augment(row['prompt'])
        augmented_scores = [1]*len(augment_result)
        if len(augment_result) < num_return_sequences:
            n_more = num_return_sequences - len(augment_result)
            augment_result += ['']*n_more
            augmented_scores+= [0]*n_more
        row['augmented_prompt'] = augment_result
        row['augmented_scores'] = augmented_scores
        return row
    
    original_df = original_df.apply(augment, axis=1)
    
    augmented_prompts = original_df['augmented_prompt'].to_list()
    augmented_prompts = list(itertools.chain.from_iterable(augmented_prompts))

    augmented_scores = original_df['augmented_scores'].to_list()
    augmented_scores = list(itertools.chain.from_iterable(augmented_scores))

    assert len(augmented_prompts) == num_return_sequences * original_df.shape[0], 'could not get enough augmented prompts'
    
    fact_id = [x for x in range(0, original_df.shape[0]) for _ in range(num_return_sequences)]
    augmented_prompts_df = pd.DataFrame({'fact_id': fact_id, 'score': augmented_scores, 'prompt': augmented_prompts, 'label': label})
    return augmented_prompts_df

def stopword_filtering(original_df: pd.DataFrame, num_return_sequences: int, label: str):
    original_prompts = original_df['prompt']
    augmented_prompts = hero.remove_stopwords(original_prompts)
    augmented_prompts = hero.remove_diacritics(augmented_prompts)
    
    augmented_prompts_df = pd.DataFrame({
        'fact_id': list(range(original_df.shape[0])),
        'score': 1,
        'prompt': augmented_prompts,
        'label': label
    })
    return augmented_prompts_df

def back_translation(original_df: pd.DataFrame, target_language: str, num_return_sequences: int, label: str):
    print(original_df.shape[0])
    sequence_per_transform = num_return_sequences * 2
    lm_src2tar = {
        'family': 'marianmt',
        'label': 'en-{}'.format(target_language),
        'model_path': 'Helsinki-NLP/opus-mt-en-{}'.format(target_language),
        'device_map': 'auto'
    }
    lm_tar2src = {
        'family': 'marianmt',
        'label': '{}-en'.format(target_language),
        'model_path': 'Helsinki-NLP/opus-mt-{}-en'.format(target_language),
        'device_map': 'auto'
    }

    beam_search_args = {
        "do_sample": False, # do greedy or greedy beam-search
        "output_scores": True,
        "return_dict_in_generate": True,
        "num_beams":  sequence_per_transform, # beam-search if >2
        "num_return_sequences": sequence_per_transform, # need to be <= num_beams
        "max_new_tokens": 100,
    }

    original_prompts = original_df['prompt'].to_list()

    tar_result = translate(
        texts=original_prompts,
        model_args=lm_src2tar,
        generation_args=beam_search_args,
        batch_size=16
    )

    ## tar to src translation
    back_result = translate(
        tar_result["texts"],
        model_args=lm_tar2src,
        generation_args=beam_search_args,
        batch_size=16
    )

    ## aggregate backtranslation
    final_texts = []
    final_scores = [] 
    back_result_per_fact_text = split_list(back_result["texts"], num_cols=sequence_per_transform ** 2)
    back_result_per_fact_score = np.reshape(back_result["scores"], [-1, sequence_per_transform])
    back_result_per_fact_score = np.reshape((back_result_per_fact_score.T * tar_result["scores"]).T, [-1, sequence_per_transform**2])
    for i in range(len(back_result_per_fact_text)):
        aggregated_backtranslation = aggregate(
            texts = back_result_per_fact_text[i],
            scores=back_result_per_fact_score[i],
            score_max= 1.0
        )
        if len(aggregated_backtranslation["texts"]) < num_return_sequences:
            n_more = num_return_sequences - len(aggregated_backtranslation["texts"])
            aggregated_backtranslation["texts"] += [""]*n_more
            aggregated_backtranslation["scores"] += [0.0]*n_more
        final_texts += aggregated_backtranslation["texts"][:num_return_sequences]
        final_scores += aggregated_backtranslation["scores"][:num_return_sequences]
    
    fact_id = [x for x in range(0, original_df.shape[0]) for _ in range(num_return_sequences)]
    augmented_prompts_df = pd.DataFrame({
        'fact_id': fact_id,
        'score': final_scores,
        'prompt': final_texts,
        'label': label
    })
    
    return augmented_prompts_df

def translate(
    texts: list,
    model_args: dict,
    generation_args: dict,
    batch_size: int
):
    model, tokenizer = get_model(model_args)
    result = generate(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        generation_args=generation_args,
        batch_size=batch_size
    )
    return result


def generate(
    texts: list,
    model,
    tokenizer,
    generation_args: dict,
    batch_size: int
):
    tokenized_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True
    )
    prompt_dataset = PromptDataset(tokenized_inputs.input_ids, tokenized_inputs.attention_mask)
    prompt_loader = DataLoader(prompt_dataset, batch_size=batch_size, shuffle=False)

    generated_tensors = []
    generation_scores = []

    model.eval()
    for _, batch in enumerate(tqdm(prompt_loader)):
        input_ids, attention_mask = batch
        output = model.generate(
            input_ids = input_ids.to(model.device),
            attention_mask = attention_mask.to(model.device),
            **generation_args
        )
        generated_tensors.append(tokenizer.batch_decode(output.sequences, skip_special_tokens=True))
        generation_scores.append(output.sequences_scores.to("cpu").detach().numpy())

    generated_tensors = list(itertools.chain.from_iterable(generated_tensors))
    generation_scores = np.exp(np.hstack(generation_scores))

    return {"texts": generated_tensors, "scores": generation_scores}


def aggregate(texts: list, scores:list, score_max=None, method = 'sum'):
    if score_max != None:
        assert isinstance(score_max, float), "score_max need to be a float"
    aggregated_result = {}
    for text, score in zip(texts, scores):
        if method == 'sum':
            aggregated_result[text] = aggregated_result.get(text, 0) + score
        elif method == 'count':
            aggregated_result[text] = aggregated_result.get(text, 0) + 1

    aggregated_result = OrderedDict(sorted(aggregated_result.items(), key=lambda x: x[1], reverse=True))

    text_list = list(aggregated_result.keys())
    score_list = list(aggregated_result.values())
    if score_max != None:
        score_list = list(score_list / score_list[0] * score_max)

    return {"texts": text_list, "scores": score_list}