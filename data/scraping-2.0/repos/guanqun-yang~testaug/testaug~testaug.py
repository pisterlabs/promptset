import os
import re
import math
import spacy
import openai
import string
import random

import pandas as pd

from tqdm import tqdm
from spacy.tokenizer import Tokenizer
from termcolor import colored
from checklist.editor import Editor
from ratelimit import limits, sleep_and_retry

from simpletransformers.classification import (
    ClassificationArgs,
    ClassificationModel
)

from utils.common import (
    seed_everything
)


seed_everything(42)

####################################################################################################
# helper function

def keep_relevant_columns(
    df, 
    columns=["capability", "description", "template", "text", "label", "pool"]
    ):
    return df[columns]


def capture_text_between_curly_braces(text):
    pattern = "\{(.*?)\}"
    texts = re.findall(pattern, text)

    texts = [text.strip() for text in texts]

    return texts


def fill_template(template, pool, n_test_case_per_template=10):
    editor = Editor()
    config = {
        "templates": template, 
        "product": True,
        "remove_duplicates": True,
        "mask_only": False,
        "unroll": False,
        "meta": False,
        "save": True,
        "labels": None,
        "nsamples": n_test_case_per_template
        }
    
    try:
        test_cases = editor.template(**config, **pool).data
    except Exception:
        print("[EXCEPTION] {} could not generate test cases".format(template))
        return
    
    return test_cases


def prepare_prompt(task="sentiment", desc=None, texts=None, label=None):
    assert task in ["sentiment", "qqp", "nli"], colored("[ERROR] Unsupported test case description", "red")

    if task == "sentiment":
        prompt = "{}\n".format(desc) + "\n".join(["- {{ {} }}\n".format(text) for text in texts])
        prompt += "\n- {"

    if task == "qqp":
        prompt = "{}\n".format(desc) + "\n".join(["- {{{{ {}  }}\n- {{ {} }}}}\n".format(ts[0], ts[1]) for ts in texts])
        prompt += "\n- {{"

    if task == "nli":
        instruction = "Write a pair of sentences that have the same relationship as the previous examples. Examples:"

        prompt = "{}\n".format(instruction) + "\n".join(["- {{{{ {}  }}\n- {{ {} }}}}\n".format(ts[0], ts[1]) for ts in texts])
        prompt += "\n- {{"

    return prompt


@sleep_and_retry
@limits(calls=50, period=60)
def query_gpt(model, prompt, n_per_query):
    response = openai.Completion.create(
                                engine=model,
                                prompt=prompt,
                                n=n_per_query,
                                max_tokens=128,
                                temperature=1,
                                top_p=1,
                                frequency_penalty=0,
                                presence_penalty=0
                                )
    return response


def capture_response(response, task="sentiment"):
    extracted_texts = list()

    if task == "sentiment":
        for choice in response.choices:
            extracted_texts.extend([t.strip("-}{" + string.whitespace) for t in choice.text.split("}")])
    
    if task in ["qqp", "nli"]:
        for choice in response.choices:
            if "\n" not in choice.text: continue

            tup = tuple([t.strip("-}{" + string.whitespace) for t in choice.text.split("\n")][:2])
            extracted_texts.append(tup)

    # keep only the string and tuple 
    extracted_texts = list(filter(lambda x: isinstance(x, str) or isinstance(x, tuple), extracted_texts))
    
    # remove empty string or tuple
    extracted_texts = list(filter(lambda x: x != "" and x != tuple(), extracted_texts))
            
    return extracted_texts


def balance_sample_by_description(df, n=1):
    # sample the df by description so that the templates will appear in both T1 and T2
    T1s, T2s = list(), list()
        
    # the description that only has 1 template
    ds = [k for k, v in (df.groupby("description").template.nunique() == 1).to_dict().items() if v]
    singleton_template = df[df.description.isin(ds)]

    T1s.append(singleton_template)
    T2s.append(singleton_template)

    n = n - len(ds)
    df = df[~df.description.isin(ds)]

    if (not df.empty) and (n > 0):
        n_unique_desc = df.description.nunique()
        if n <= n_unique_desc:
            ts = random.sample(df.groupby("description").template.sample(n=1, random_state=42).tolist(), n)
        else:
            k = math.floor(n / n_unique_desc)
            max_size = df.groupby("description").template.nunique().max()

            if k > max_size: k = max_size

            ts = df.groupby("description").template.sample(n=k, random_state=42).tolist()
            ts += list(random.sample(set(df.template.tolist()) - set(ts), int(n - k * n_unique_desc)))

        T2s.append(df[df.template.isin(ts)])
        T1s.append(df[~df.template.isin(ts)])

    T2 = pd.concat(T2s)
    T1 = pd.concat(T1s)

    return T1, T2

####################################################################################################


def generate_template_test_suite(T, n_test_case_per_template=10, template_column="template"):
    if T.empty: return
    assert template_column in T.columns

    # make sure the templates do not duplicate
    T = T.drop_duplicates(subset="template")

    records = list()
    for _, t in T.iterrows():
        # NOTE: the original "text" field in t is overridden by the newly generated text
        records.extend(
            {
                **t.to_dict(),
                "text": text
            }
            for text in fill_template(t[template_column], t["pool"], n_test_case_per_template=n_test_case_per_template)
        )
        
    df = pd.DataFrame(records).drop_duplicates(subset="text").reset_index(drop=True)
    return df
    

def generate_gpt3_test_suite(
        T, 
        task="sentiment",
        model="text-davinci-001",
        n_demonstration=3, 
        n_per_query=10,
        n_test_case_per_template=100
    ):
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    seed_df = T

    records = list()
    # query GPT for each template
    for _, row in seed_df.groupby("template").agg(list).reset_index().iterrows():
        template = row["template"]

        # make sure there are at most n_demonstration demonstrations
        demonstrations = row["text"][:n_demonstration]

        # the "description" and "label" will be duplicated for n_demonstration times
        # but we only need to take the first one
        cap = row["capability"][0]
        desc = row["description"][0]
        label = row["label"][0]
        # the NLI dataset does not have a pool field so assign None to this field
        pool = row["pool"][0] if "pool" in row else None

        prompt = prepare_prompt(task=task, desc=desc, texts=demonstrations, label=label)
        unique_texts = set()

        query_cnt = 0
        query_budget = int(n_test_case_per_template / n_per_query * 2)
        with tqdm(total=n_test_case_per_template) as pbar:
            while len(unique_texts) < n_test_case_per_template:

                response = query_gpt(model=model, prompt=prompt, n_per_query=n_per_query)
                texts = capture_response(response, task=task)
            
                # make sure the response does not repeat the demonstrations
                texts = set(texts) - set(demonstrations)
                if not texts: continue

                # check the progress of unique generations
                pbar.update(len(texts - unique_texts))

                unique_texts |= texts
                query_cnt += 1

                # query GPT-3 2 times more than necessary to obtain enough sentences
                if query_cnt >= query_budget: break

        records.extend(
            {   
                "capability": cap,
                "description": desc,
                "template": template,
                "text": text,
                "label": label,
                "prompt": prompt,
                "demonstration": demonstrations,
                "pool": pool
            } for text in unique_texts
        )

    return pd.DataFrame(records)
            

def generate_new_templates(df, verbose=False):
    slot_tags = ["VERB", "NOUN", "ADJ"]

    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)

    # keep relevant columns only
    df = keep_relevant_columns(df)

    # collapse with same columns
    df = df.groupby(["capability", "description", "template", "label"]).agg(list).reset_index()

    records = list()
    with tqdm(total=len(df), disable=not verbose) as pbar:
        for _, row in df.iterrows():
            pbar.update(1)

            pool = row["pool"][0]
            texts = row["text"]

            cap = row["capability"]
            desc = row["description"]
            template = row["template"]
            label = row["label"]

            # create word: slot type dictionary
            slot_dict = dict()
            for k, vs in pool.items():
                if not all([isinstance(v, str) for v in vs]): continue

                for v in vs:
                    if len(v.split()) > 1: continue
                    slot_dict[v] = "{{{}}}".format(k)

            if slot_dict == dict(): continue

            # create new templates
            for gpt_text in texts:
                # single sentence
                if isinstance(gpt_text, str):
                    doc = nlp(gpt_text)

                    new_template = " ".join([slot_dict.get(token.text, token.text) if token.pos_ in slot_tags else token.text for token in doc])
                    if capture_text_between_curly_braces(new_template) == list(): continue

                # two sentence
                if isinstance(gpt_text, tuple):
                    new_templates = list()
                    for gpt_text_sent in gpt_text:
                        doc = nlp(gpt_text_sent)

                        new_template_sent = " ".join([slot_dict.get(token.text, token.text) if token.pos_ in slot_tags else token.text for token in doc])
                        if capture_text_between_curly_braces(new_template_sent) == list(): continue

                        new_templates.append(new_template_sent)
                    new_template = tuple(new_templates)

                if new_template == template: continue

                records.append({
                    "capability": cap,
                    "description": desc,
                    "template": template,
                    "new_template": new_template,
                    "label": label,
                    "pool": pool
                })

    df = pd.DataFrame(records).drop_duplicates(subset="new_template").reset_index(drop=True)
    return df


def prepare_testing(df_dict):
    for name in list(df_dict.keys()):
        df = df_dict[name]
        if (df is None) \
           or (isinstance(df, pd.DataFrame) and df.empty): 
           del df_dict[name]

    n = min([len(df) for df in df_dict.values()])
    for name in df_dict.keys(): 
        df_dict[name] = df_dict[name].sample(n=n, random_state=42).assign(source=name)

    patch_dfs, test_dfs = list(), list()
    for df in df_dict.values():
        n_unique_template = df.template.nunique()

        sample_size = int(n_unique_template / 2) if n_unique_template % 2 == 0 else int((n_unique_template + 1) / 2)
        T1, T2 = balance_sample_by_description(df, n=sample_size)

        patch_dfs.append(T1)
        test_dfs.append(T2)

    patch_df = pd.concat(patch_dfs)
    test_df = pd.concat(test_dfs)

    return patch_df, test_df


def process_sentence_pair_dataset(df):
    df = pd.DataFrame(
        [
            {
                "text_a": tup.text[0],
                "text_b": tup.text[1],
                "labels": tup.labels
            }
            for tup in df.itertuples()
        ]
    )
    return df


def test_model(model_name, model_type, task, num_labels, patch_df, test_df):
    # in case there are empty inputs, this should work for both string and tuple
    patch_df = patch_df[patch_df.text.apply(lambda x: len(x) > 1)].reset_index(drop=True)
    test_df = test_df[test_df.text.apply(lambda x: len(x) > 1)].reset_index(drop=True)

    model_args = ClassificationArgs()

    model_args.manual_seed = 42

    model_args.max_seq_length = 128
    model_args.train_batch_size = 2
    model_args.eval_batch_size = 2

    model_args.num_train_epochs = 3
    model_args.save_model_every_epoch = False
    model_args.save_steps = -1
    model_args.evaluate_during_training = False

    model_args.output_dir = "testing"
    model_args.best_model_dir = "testing/best_model"
    model_args.tensorboard_dir = "runs/testing"
    model_args.overwrite_output_dir = True

    model = ClassificationModel(
        model_type,
        model_name,
        num_labels=num_labels,
        args=model_args,
    )

    test_texts = test_df.text.tolist()


    # process dataset into format suitable for simpletransformers to process
    patch_df = patch_df.rename(columns={"label": "labels"})
    if task in ["qqp", "nli"]:
        patch_df = process_sentence_pair_dataset(patch_df)

    # before patching
    before, _ = model.predict(test_texts)
    test_df = test_df.assign(before=before)

    # patching
    model.train_model(
        patch_df,
        eval_df=None
    )

    # after patching
    after, _ = model.predict(test_texts)
    test_df = test_df.assign(after=after)

    return test_df

    




