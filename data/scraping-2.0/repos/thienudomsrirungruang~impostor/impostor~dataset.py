from typing import *

import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import OpenAIGPTTokenizer

from itertools import chain
from collections import defaultdict

from special_tokens import bos, eos, speaker_self, speaker_other, lsep, pad, SPECIAL_TOKENS

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device: {}".format(device))


def build_inputs(history: List[Tuple[bool, List[str]]], reply: Optional[Tuple[bool, List[str]]],
                 tokenizer: transformers.OpenAIGPTTokenizer, populate_lm_labels=False, with_eos=True, with_reply=True):
    if with_reply:
        history = history + [reply]
    sequence = list(map(lambda x: [speaker_self if x[0] else speaker_other] + x[1], history))
    # print(sequence)
    sequence[0] = [bos] + sequence[0]
    if with_eos:
        sequence[-1] = sequence[-1] + [eos]
    words = list(chain(*sequence))
    segments = list(chain(*[[speaker_self if s[0] else speaker_other] * len(sequence[i]) for i, s in enumerate(history)]))
    input_ids = tokenizer.convert_tokens_to_ids(words)
    mc_token_ids = len(input_ids) - 1
    token_type_ids = tokenizer.convert_tokens_to_ids(segments)
    lm_labels = [-100] * len(input_ids)
    if populate_lm_labels:
        lm_labels = ([-100] * sum(len(s) for s in sequence[:-1])) + tokenizer.convert_tokens_to_ids(sequence[-1])
    return input_ids, mc_token_ids, token_type_ids, lm_labels


class ChatDataset(Dataset):
    def __init__(self, dataset_object: DefaultDict, tokenizer: transformers.OpenAIGPTTokenizer):
        self._dataset_object = dataset_object
        self._length = len(dataset_object["history"])
        self._tokenizer = tokenizer

    def __getitem__(self, idx: int) -> DefaultDict:
        out = defaultdict(list)
        tokenizer = self._tokenizer
        history = list(map(lambda x: (x[0], tokenizer.tokenize(x[1])), self._dataset_object["history"][idx]))
        correct = self._dataset_object["correct"][idx]
        out["correct"] = correct
        for i, candidate in enumerate(self._dataset_object["candidates"][idx]):
            candidate = (candidate[0], tokenizer.tokenize(candidate[1]))
            input_ids, mc_token_ids, token_type_ids, lm_labels = build_inputs(history, candidate, tokenizer, i == correct)
            # cutoff at -500 to prevent overflow
            out["input_ids"].append(input_ids[-500:])
            out["mc_token_ids"].append(mc_token_ids)
            out["token_type_ids"].append(token_type_ids[-500:])
            out["lm_labels"].append(lm_labels)
        return out

    def __len__(self):
        return self._length


def pad_list(x: List[int], padding: int, padding_length: int):
    return x + [padding] * (padding_length - len(x))


def make_batch(dialogs: List[DefaultDict], pad_token_id: int):
    out = {}
    max_length = max(max(len(y) for y in x["input_ids"]) for x in dialogs)
    for k, v in dialogs[0].items():
        if k == "correct":
            out[k] = torch.tensor([x["correct"] for x in dialogs], dtype=torch.long)
        elif k == "mc_token_ids":
            out[k] = torch.tensor([x["mc_token_ids"] for x in dialogs], dtype=torch.long)
        else:
            out[k] = torch.tensor([[pad_list(y, -100 if k == "lm_labels" else pad_token_id, max_length) for y in x[k]] for x in dialogs],
                                  dtype=torch.long)
    return out


def get_data_loader(dataset: ChatDataset, tokenizer: transformers.OpenAIGPTTokenizer,
                    batch_size: int = 4, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                        collate_fn=lambda x: make_batch(x, pad_token_id))
    return loader


def get_dataset(dataset_path: str, tokenizer: transformers.OpenAIGPTTokenizer):
    dataset_object = torch.load(dataset_path)
    dataset = ChatDataset(dataset_object, tokenizer)
    return dataset


if __name__ == "__main__":
    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS)

    dataset = get_dataset("../dataset/testing-set.pt", tokenizer)

    loader = get_data_loader(dataset, tokenizer)
    for x in loader:
        print(x)
