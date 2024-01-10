from typing import *

import torch
from transformers import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer

from itertools import chain

from special_tokens import bos, eos, speaker_self, speaker_other, lsep, pad, SPECIAL_TOKENS

model = OpenAIGPTDoubleHeadsModel.from_pretrained("openai-gpt")
tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

# history = [[(True, "hello"), (True, "how"), (True, "are"), (True, "you"), (True, "?")],
#            [(False, "i"), (False, "am"), (False, "fine"), (False, "thanks"), (False, ".")]]

history = [(True, tokenizer.tokenize("hello how are you?")),
           (False, tokenizer.tokenize("i am fine thanks."))]

reply = (True, ["good", "to", "hear", "."])

orig_num_tokens = len(tokenizer.encoder)
print(orig_num_tokens)
num_added_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS)
model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def build_inputs(history: List[Tuple[bool, List[str]]], reply: Tuple[bool, List[str]]):
    history = history + [reply]
    sequence = list(map(lambda x: [speaker_self if x[0] else speaker_other] + x[1], history))
    # print(sequence)
    sequence[0] = [bos] + sequence[0]
    sequence[-1] = sequence[-1] + [eos]
    # print(sequence)
    words = list(chain(*sequence))
    segments = list(chain(*[[speaker_self if s[0] else speaker_other] * len(sequence[i]) for i, s in enumerate(history)]))
    position = list(range(len(words)))
    return words, segments, position, sequence


words, segments, position, sequence = build_inputs(history, reply)

print(words, segments, position, sequence, sep="\n")

words = tokenizer.convert_tokens_to_ids(words)
segments = tokenizer.convert_tokens_to_ids(segments)

print("Actual:")
print(words, segments, sep="\n")

# example data
distractor = (True, tokenizer.tokenize("sorry to hear that :("))

words_distractor, segments_distractor, _, _ = build_inputs(history, distractor)
words_distractor = tokenizer.convert_tokens_to_ids(words_distractor)
segments_distractor = tokenizer.convert_tokens_to_ids(segments_distractor)
print("Distractor:")
print(words_distractor, segments_distractor, sep="\n")


lm_targets = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + tokenizer.convert_tokens_to_ids(sequence[-1][1:])
lm_distractor = [-100] * len(words_distractor)

last_token = len(words) - 1
last_token_distractor = len(words_distractor) - 1

padding_length = max(len(words), len(words_distractor))


def pad(x: List[int], padding: int):
    return x + [padding] * (padding_length - len(x))


words, words_distractor, segments, segments_distractor = [pad(x, tokenizer.convert_tokens_to_ids("<pad>"))
                                                          for x in (words, words_distractor, segments, segments_distractor)]

lm_targets, lm_distractor = [pad(x, -100) for x in (lm_targets, lm_distractor)]

print("Actual:", words, segments, lm_targets, sep="\n")
print("Distractor:", words_distractor, segments_distractor, lm_distractor, sep="\n")

input_ids = torch.tensor([[words, words_distractor]], dtype=torch.long)
token_type_ids = torch.tensor([[segments, segments_distractor]], dtype=torch.long)
mc_token_ids = torch.tensor([[last_token, last_token_distractor]], dtype=torch.long)
lm_labels = torch.tensor([[lm_targets, lm_distractor]], dtype=torch.long)
mc_labels = torch.tensor([0], dtype=torch.long)

print("input_ids: {}\ntoken_type_ids: {}\nmc_token_ids: {}\nlm_labels: {}\nmc_labels: {}"
      .format(input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels))

model_output = model(input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids, mc_labels=mc_labels, labels=lm_labels)

print(model_output.loss.item(), model_output.mc_loss.item())
