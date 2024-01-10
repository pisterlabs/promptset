### Sentence Splitter Class
from src.sentence_split_prompt import SPLIT_PROMPT, SYSTEM_PROMPT, USER_PROMPT
import nltk
import openai
import string
from typing import List
import backoff

nltk.download('punkt')


class SentenceSplitter:
    def __init__(self, dummy=True):
        self.dummy = dummy
        openai.api_key = os.getenv('OPENAI_API_KEY')
        return
    
    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def _split_(self, system_prompt, user_prompt):  # prompt):
        # return openai.Completion.create(
        #     model="text-davinci-003",
        #     prompt=prompt,
        #     max_tokens=256,
        #     n=1
        # )['choices'][0]
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=256,
            temperature=0.5,
            n=1
        )['choices'][0].message
    
    def split(self, complex_sent: str) -> List[str]:
        if self.dummy:
            return [f"{' '.join(['dummy'] * 10)}."] * 3
        # response = self._split_(SPLIT_PROMPT.format(complex_sent))
        response = self._split_(SYSTEM_PROMPT, USER_PROMPT.format(complex_sent))
        # simple_sents_list_unprocessed = response['text'].split('\n')
        simple_sents_list_unprocessed = response['content'].split('\n')
        simple_sents_list = []
        for simple_sent in simple_sents_list_unprocessed:
            if simple_sent[:2] == '- ':
                simple_sent = simple_sent[2:].strip(" ")
                simple_sents_list.append(simple_sent)
        if len(simple_sents_list) == 0 and len(words:=nltk.word_tokenize(complex_sent.translate(str.maketrans('', '', string.punctuation)))) > 0:
                return [complex_sent]
        else:
            return simple_sents_list


import os
import openai
from src.sentence_split_prompt import SPLIT_PROMPT
import json
from transformers import GPT2TokenizerFast
import nltk
from typing import List
from src.create_dataset import check_written_dataset

nltk.download('punkt')

required_headers = {
    # 'source_reviews_a',
    # 'source_reviews_b',
    'refs_a',
    'refs_b',
    'refs_comm',
    # 'gen_a',
    # 'gen_b',
    # 'gen_comm',
}


def _get_summ_sent_dict(summ: str, example_no: int, text_type: str, text_no: int):
    text_type_to_prefix = {
        'refs_a': 'RA',
        'refs_b': 'RB',
        'refs_comm': 'RC',
        'gen_a': 'GA',
        'gen_b': 'GA',
        'gen_comm': 'GC',
        'source_reviews_a': 'SA',
        'source_reviews_b': 'SB'
    }
    summ_sent_list = nltk.sent_tokenize(summ)
    # summ_sent_list = summ.split(".")
    summ_sent_dict = {}
    for sent_no, sent in enumerate(summ_sent_list):
        sent_id = f"E{example_no:03d}{text_type_to_prefix[text_type]}{text_no:02d}N{sent_no:03d}"
        summ_sent_dict[sent_id] = sent
    return summ_sent_dict


def get_sent_indexed_dataset(source_dataset):
    new_data = []
    for example_id, example in enumerate(source_dataset):
        new_example = example
        for header in example:
            if header not in required_headers:
                new_example[header] = example[header]
                continue
            old_header_data = example[header]
            if type(old_header_data) is list:
                new_header_data = []
                for idx, old_value in enumerate(old_header_data):
                    new_value = _get_summ_sent_dict(old_value, example_id, header, idx)
                    new_header_data.append(new_value)
            else:
                old_header_data = [example[header]]
                for old_value in old_header_data:
                    new_header_data = _get_summ_sent_dict(old_value, example_id, header, 0)

            new_example.update({header: new_header_data})
        new_data.append(new_example)

    return new_data


def get_paragraph_simple_sent_dataset(source_dataset):
    new_data = []
    for example_no, example in enumerate(source_dataset):
        for header in required_headers:
            split_paragraph_dicts = example[header]
            paragraphs = []
            for split_paragraph_dict in split_paragraph_dicts:
                complex_sents = []
                for sent_id, sent_dict in split_paragraph_dict.items():
                    simple_sents = list(sent_dict['simple_sents'].values())
                    complex_sent = " ".join(simple_sents)
                    complex_sents.append(complex_sent)
                paragraph = " ".join(complex_sents)
                paragraphs.append(paragraph)
            example[header] = paragraphs
        new_data.append(example)
    return new_data


def get_indexed_simple_sent_dataset(source_dataset, temp_folder_prefix=""):
    if not os.path.isfile("data/temporary_dataset_files/" + temp_folder_prefix + "completed_split_sent_ids.json"):
        completed_sent_ids = {}
    else:
        completed_sent_ids = json.load(open("data/temporary_dataset_files/" + temp_folder_prefix + "completed_split_sent_ids.json", 'r'))
    # print(len(sent_indexed_dataset))
    # print(json.dumps(sent_indexed_dataset[:2], indent=4))
    sentence_splitter = SentenceSplitter(dummy=False)
    simple_sent_dataset = []
    for example_no, example in enumerate(sent_indexed_dataset):
        # print(json.dumps(example, indent=4))
        for header in required_headers:
            source_is_list = type(example[header]) is list
            if source_is_list and len(example[header]) == 8:
                paragraphs = example[header]
            elif source_is_list:
                paragraphs = [example[header][0]]
            else:
                paragraphs = [example[header]]
            # print("".join(['*']*23),"PARAGRAPHS","".join(['*']*23),)
            # print(json.dumps(paragraphs, indent=4))
            split_paragraph_dicts = []
            for paragraph_no, paragraph in enumerate(paragraphs):
                # print(f"PARAGRAPH NO. {paragraph_no:02d}")
                # print(json.dumps(paragraph, indent=4))
                new_paragraph = {}
                # print(paragraph)
                for sent_id, sent in paragraph.items():
                    if sent_id in completed_sent_ids:
                        new_paragraph[sent_id] = {
                            'sentence': sent,
                            'simple_sents': completed_sent_ids[sent_id]
                        }
                        continue
                    # print(f"{sent_id} = {sent}")
                    simple_sents = sentence_splitter.split(sent)
                    simple_sents_ids = [f"{sent_id}_{idx:03d}" for idx in range(0, len(simple_sents))]
                    simple_sent_dict = dict(zip(simple_sents_ids, simple_sents))
                    completed_sent_ids[sent_id] = simple_sent_dict
                    json.dump(completed_sent_ids, open("data/temporary_dataset_files/" + temp_folder_prefix + "completed_split_sent_ids.json", 'w'))
                    new_paragraph[sent_id] = {
                        'sentence': sent,
                        'simple_sents': simple_sent_dict
                    }
                split_paragraph_dicts.append(new_paragraph)
            # print("".join(['*']*23),f"SPLIT PARAGRAPH DICTS","".join(['*']*23),)
            # print(json.dumps(split_paragraph_dicts, indent=4))
            example[header] = split_paragraph_dicts
            # print("reaching here")
            # break
        # break
        simple_sent_dataset.append(example)
    return simple_sent_dataset



## To convert paraphrase data
# # source reviews are the same across all datasets (base, paraphrase, selfparaphrase)
# original_dataset = json.load(open("data/temporary_dataset_files/old_files/combined_data_base_paraphrase_sent_level.json", 'r'))

# sent_indexed_dataset = get_sent_indexed_dataset(original_dataset)
# json.dump(sent_indexed_dataset, open("data/temporary_dataset_files/paraphrase_split_sent_level/sent_indexed_dataset.json", "w"))

# indexed_split_sent_dataset = get_indexed_simple_sent_dataset(sent_indexed_dataset, temp_folder_prefix="paraphrase_split_sent_level/")
# json.dump(indexed_split_sent_dataset, open("data/temporary_dataset_files/paraphrase_split_sent_level/indexed_split_sent_dataset.json", "w"))

# paragraph_split_sent_dataset = get_paragraph_simple_sent_dataset(indexed_split_sent_dataset)
# json.dump(paragraph_split_sent_dataset, open("data/temporary_dataset_files/paraphrase_split_sent_level/combined_data_paraphrase_split_sent_level.json", "w"))

## To convert similarity dataset
# source reviews are the same across all datasets (base, paraphrase, selfparaphrase) but in 
# similarity the source review sets of both entities are the same

# original_dataset = json.load(open("data/combined_data_selfparaphrase.json", 'r'))

# sent_indexed_dataset = get_sent_indexed_dataset(original_dataset)
# json.dump(sent_indexed_dataset, open("data/temporary_dataset_files/selfparaphrase_sent_level/sent_indexed_dataset.json", "w"))

# indexed_split_sent_dataset = get_indexed_simple_sent_dataset(sent_indexed_dataset, temp_folder_prefix="selfparaphrase_sent_level/")
# json.dump(indexed_split_sent_dataset, open("data/temporary_dataset_files/selfparaphrase_sent_level/indexed_split_sent_dataset.json", "w"))

# paragraph_split_sent_dataset = get_paragraph_simple_sent_dataset(indexed_split_sent_dataset)
# json.dump(paragraph_split_sent_dataset, open("data/combined_data_selfparaphrase_split_complete.json", "w"))

original_dataset = json.load(open("data/combined_data_paraphrase_synonyms.json", 'r'))

sent_indexed_dataset = get_sent_indexed_dataset(original_dataset)
json.dump(sent_indexed_dataset, open("data/temporary_dataset_files/paraphrase_synonyms/sent_indexed_dataset.json", "w"))

indexed_split_sent_dataset = get_indexed_simple_sent_dataset(sent_indexed_dataset[26:], temp_folder_prefix="paraphrase_synonyms/")
json.dump(indexed_split_sent_dataset, open("data/temporary_dataset_files/paraphrase_synonyms/indexed_split_sent_dataset.json", "w"))

paragraph_split_sent_dataset = get_paragraph_simple_sent_dataset(indexed_split_sent_dataset)
json.dump(paragraph_split_sent_dataset, open("data/combined_data_paraphrase_synonyms_split_complete.json", "w"))