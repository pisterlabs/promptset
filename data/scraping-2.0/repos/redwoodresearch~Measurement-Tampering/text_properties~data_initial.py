import json
import os
import random
import re
import threading
import time
from collections import defaultdict

import attrs
import openai
import tiktoken
from datasets import load_dataset

from text_properties.input_text_properties import QueryItem, QueryItems, cut_text, data_to_queries, get_completions
from text_properties.modify_text import RewriteItem, get_rewrites
from text_properties.properties import (
    all_input_properties,
    all_output_properties,
    general_properties,
    latent_output_properties,
    output_only_properties,
    sensors,
)
from text_properties.sensor_query import get_sensor_query

# %%

tokenizer = tiktoken.encoding_for_model("gpt-4")

if "OPENAI_API_KEY" not in os.environ:
    openai.api_key_path = os.path.expanduser("~/.openai_api_key")


# %%

ds = load_dataset("EleutherAI/pile", streaming=True, split="train")

# %%

N = 5_000

# %%

# N = 10

data = list(ds.take(N))
random.shuffle(data)


# %%

with open("temp_out_text.txt", "w") as f:
    f.write(data[23]["text"])


# %%


query_items, _ = data_to_queries(data[:2000], all_input_properties)

print(len(tokenizer.encode(query_items[4].query_unwrap[-1]["content"])), len(query_items))


# %%

with open("out_content.jsonl", "w") as f:
    for q in query_items[:5]:
        f.write(json.dumps(q.query) + "\n")

# %%

with open("out_content.jsonl", "r") as f:
    items = [json.loads(l) for l in f.read().splitlines()]


# print([3]["content"])

# %%


# %%

out_file = "fixed_new.jsonl"

with open(out_file, "w") as f:
    ...

n_threads = max(min(len(query_items), 30), 1)

shuffle_items = list(query_items)
random.shuffle(shuffle_items)

threads = [
    threading.Thread(
        target=get_completions,
        args=(
            shuffle_items,
            k,
            n_threads,
            ["gpt-3.5-turbo-16k"],  # for higher rate limit lol
            out_file,
        ),
    )
    for k in range(n_threads)
]

# %%


for t in threads:
    t.start()

# %%

for t in threads:
    t.join()

# %%

print("done!")

# %%

with open("fixed_new.jsonl", "r") as f:
    loaded_items = [json.loads(l) for l in f.read().splitlines()]
len(loaded_items)

# %%

# all_input_properties = [x.partition("Does the text contain ")[-1][:-1] for x in  loaded_items[0]["query"]["query"][-1]["content"].splitlines()[2:]]

# %%


@attrs.frozen
class ParsedResponse:
    query: QueryItem
    answers_by_model: dict[str, list[bool]]

    @classmethod
    def parse_out(cls, x):
        by_model = {}
        for model, rest in x["response"].items():
            response_lines = rest["choices"][0]["message"]["content"].splitlines()
            assert len(response_lines) == len(all_input_properties)
            by_model[model] = [y.split()[-1] == "Yes" for y in response_lines]
        return cls(QueryItem(**x["query"]), by_model)


# %%

# len(all_input_properties)
# for x in list(
#     zip(
#         loaded_items[0]["response"]["gpt-3.5-turbo-16k"]["choices"][0]["message"]["content"].splitlines(),
#         all_input_properties,
#     )
# ):
#     print(x)
# #
# # len(loaded_items[0]["response"]["gpt-3.5-turbo-16k"]["choices"][0]["message"]["content"].splitlines())

# %%

parsed_responses = []
for x in loaded_items:
    try:
        parsed_responses.append(ParsedResponse.parse_out(x))
    except AssertionError:
        ...

len(parsed_responses)

# %%


def print_item(x: ParsedResponse):
    response_str = "\n".join(
        f"{question} {gpt4_item} {gpt3_5_item}" + (" DISAGREE" if gpt4_item != gpt3_5_item else "")
        for gpt4_item, gpt3_5_item, question in zip(
            x.answers_by_model["gpt-4-0613"], x.answers_by_model["gpt-3.5-turbo-0613"], all_input_properties
        )
    )

    return x.query.cut_text + "\n" + response_str


# print(print_item(parsed_responses[0]))

# %%


def basic_print_str(x: ParsedResponse):
    return (
        x.query.cut_text
        + "\n\n"
        + "\n".join(
            f"{question}: {ans}" for question, ans in zip(all_input_properties, x.answers_by_model["gpt-3.5-turbo-16k"])
        )
    )


# print()
# print(basic_print_str(parsed_responses[16]))

# %%

import numpy as np

sum_count = 0

for resp in parsed_responses:
    sum_count += np.array(resp.answers_by_model["gpt-3.5-turbo-16k"])

# %%

s = "\n".join(
    f"{question}: {c} ({c / len(parsed_responses):.5f})" for question, c in zip(all_input_properties, sum_count)
)
print(len(parsed_responses))
print(s)

# %%


# %%


# %%

writing_query_items: list[RewriteItem] = []

for d in data[260:500]:
    # for d in data[200:250]:
    new_text = cut_text(d["text"])
    if new_text is None:
        continue
    out_lines = new_text.splitlines()

    copied = list(all_output_properties)
    random.shuffle(copied)

    new_properties = latent_output_properties + copied[:2]
    random.shuffle(new_properties)

    writing_query_items.append(
        RewriteItem(
            new_text,
            new_properties,
            d["text"],
        )
    )

len(writing_query_items)

# %%

with open("writing_query_items.jsonl", "w") as f:
    for q in writing_query_items:
        f.write(json.dumps(attrs.asdict(q)) + "\n")

# %%


with open("writing_query_items.jsonl", "r") as f:
    queries_load = [json.loads(l) for l in f.read().splitlines()]

# %%

# out_file = "writing_out_new.jsonl"
out_file = "writing_out_new_3_5.jsonl"

with open(out_file, "w") as f:
    ...

n_threads = max(min(len(writing_query_items), 80), 1)

shuffle_items = list(writing_query_items)
random.shuffle(shuffle_items)

threads = [
    threading.Thread(
        target=get_rewrites,
        args=(
            shuffle_items,
            k,
            n_threads,
            ["gpt-4-0613"],
            # ["gpt-3.5-turbo-16k"],
            out_file,
            1500,
            0.3,
        ),
    )
    for k in range(n_threads)
]

# %%

for t in threads:
    time.sleep(0.1)
    t.start()

# %%

for t in threads:
    t.join()

# %%

print("done!")

# %%

with open("writing_out_new.jsonl", "r") as f:
    # with open("writing_out_new_3_5.jsonl", "r") as f:
    loaded_items_writing = [json.loads(l) for l in f.read().splitlines()]
len(loaded_items_writing)

# %%

mods = loaded_items_writing[2]["query"]["modifications"]
model = loaded_items_writing[2]["model"]

# %%

with open("temp_out_text.txt", "w") as f:
    f.write(loaded_items_writing[2]["all_text"][6])


# %%


@attrs.frozen
class WritingResponse:
    query: RewriteItem
    model: str
    final_text: str
    all_text: list[str]

    @classmethod
    def parse_out(cls, x):
        return cls(RewriteItem(**x["query"]), model=x["model"], final_text=x["all_text"][-1], all_text=x["all_text"])


# %%

all_writing_responses: list[WritingResponse] = []

for item in loaded_items_writing:
    try:
        all_writing_responses.append(WritingResponse.parse_out(item))
    except AssertionError:
        ...

# %%

all_writing_responses[1].query.modifications
all_writing_responses[1].model

with open("temp_out_text.txt", "w") as f:
    f.write(all_writing_responses[1].all_text[4])

# %%


# %%

sensor_query_items: QueryItems = []

for resp in all_writing_responses:
    for sensor in sensors:
        sensor_query_items.append(
            QueryItem(
                resp.query.cut_text,
                get_sensor_query(sensor, resp.final_text),
                resp.query.full_text,
                extra=dict(sensor=sensor, writing_response=resp),
            )
        )

len(sensor_query_items)

# %%


out_file = "sensors_out.jsonl"
# out_file = "sensors_out_3_5.jsonl"

with open(out_file, "w") as f:
    ...

n_threads = max(min(len(sensor_query_items), 1), 1)

shuffle_items = list(sensor_query_items)
random.shuffle(shuffle_items)

threads = [
    threading.Thread(
        target=get_completions,
        args=(
            shuffle_items,
            k,
            n_threads,
            # ["gpt-4-0613"],
            ["gpt-3.5-turbo-16k"],
            out_file,
            50,
            dict(
                functions=[
                    {
                        "name": "provide_answer",
                        "description": "Answers the question with Yes/No",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                    "description": "The Yes/No answer",
                                    "enum": ["Yes", "No"],
                                },
                            },
                        },
                        "required": ["answer"],
                    }
                ],
                function_call={"name": "provide_answer"},
            ),
        ),
    )
    for k in range(n_threads)
]

# %%

for t in threads:
    t.start()

# %%

for t in threads:
    t.join()

# %%

print("done!")

# %%

with open("sensors_out.jsonl", "r") as f:
    # with open("sensors_out_3_5.jsonl", "r") as f:
    loaded_items_sensors = [json.loads(l) for l in f.read().splitlines()]
len(loaded_items_sensors)

# %%

sensor_values: defaultdict[str, defaultdict[str, defaultdict[str, dict[str, bool]]]] = defaultdict(
    lambda: defaultdict(lambda: defaultdict(dict))
)

sensor_values_by_passage: defaultdict[tuple[str, str, str], dict[str, bool]] = defaultdict(dict)

for sensor_item in loaded_items_sensors:
    for model, item in sensor_item["response"].items():
        which_model_wrote = sensor_item["query"]["extra"]["writing_response"]["model"]
        which_sensor = sensor_item["query"]["extra"]["sensor"]
        sensor_val = json.loads(item["choices"][0]["message"]["function_call"]["arguments"])["answer"] == "Yes"
        sensor_values[sensor_item["query"]["cut_text"]][which_model_wrote][model][which_sensor] = sensor_val
        sensor_values_by_passage[
            sensor_item["query"]["extra"]["writing_response"]["final_text"],
            which_model_wrote,
            model,
        ][which_sensor] = sensor_val

# %%

# sensor_item_filter = {
#     k: v for k, v in sensor_values_by_passage.items() if k[1] == "gpt-4-0613" and k[2] == "gpt-4-0613"
# }
sensor_item_filter = sensor_values_by_passage

some_neg = {k: v for k, v in sensor_item_filter.items() if any(not x for x in v.values())}

len(some_neg) / len(sensor_item_filter)

# %%

all_b = [x for xs in sensor_item_filter.values() for x in xs.values()]
(sum(all_b) / len(all_b)) ** 5

# %%

count = 0
for q in query_items:
    count += re.search(r"\b[hH]e\b", q.cut_text) is not None
count / len(query_items)

# %%

count = 0
for q in query_items:
    count += q.cut_text.count(".") > 10
count / len(query_items)


# %%


count = 0
for q in query_items:
    count += (len(re.findall(r"\bI\b", q.cut_text)) - q.cut_text.count(".")) > -7
count / len(query_items)


# %%

xs = general_properties + output_only_properties
xs[:50]

# xs = list(input_latent_properties)
# random.shuffle(xs)

# %%


latent_props = [(p, c) for p, c in zip(all_input_properties, sum_count) if 0.1 < c / len(parsed_responses) < 0.8]

# %%

# latent_props = [(p, c) for p, c in zip(all_input_properties, sum_count) if c / len(parsed_responses) > 0.6]

# %%

latent_props

# %%

props_to_sample = [p for p, c in zip(all_input_properties, sum_count) if c / len(parsed_responses) > 0.005]
random.shuffle(props_to_sample)
len(props_to_sample)

# %%

with open("elk/text_properties/most_common_words.txt", "r") as f:
    common_words = [x.strip() for x in f.readlines()[150:250]]

# %%

regexes = {w: re.compile(rf"\b{w}\b", re.IGNORECASE) for w in common_words}
counts = defaultdict(int)

for q in query_items[:500]:
    for w, regex in regexes.items():
        counts[w] += regex.search(q.cut_text) is not None

# %%


with open("elk/text_properties/most_common_words.txt", "r") as f:
    common_words_many = [x.strip() for x in f.readlines()[50:150]]

regexes_many = {w: re.compile(rf"\b{w}\b", re.IGNORECASE) for w in common_words_many}
all_word_occur: list[list[bool]] = []

for q in query_items[:500]:
    all_word_occur.append([regex.search(q.cut_text) is not None for w, regex in regexes_many.items()])

all_occur = np.array(all_word_occur)
fracs_oc = all_occur.mean(axis=0)
oc_vars_v = fracs_oc - fracs_oc**2
oc_stds_v = np.sqrt(oc_vars_v)

cov_sum = 0
diff_sum = 0
for vals in all_occur:
    cov_sample = (vals[:, None] - fracs_oc) * (vals[None, :] - fracs_oc)
    dif_vals = np.maximum(vals[:, None].astype(np.float64) - vals[None, :], 0.0)
    cov_sum += cov_sample
    diff_sum += dif_vals

cov = cov_sum / len(all_word_occur)
diff = diff_sum / len(all_word_occur)
diff_norm = diff / fracs_oc[:, None]

corr = cov / (oc_stds_v[:, None] * oc_stds_v[None, :])


all_pairs = (corr > 0.20) & (diff_norm > 0.05) & (diff_norm > 0.05).T

i_arr, j_arr, k_arr = np.nonzero(all_pairs[:, :, None] & all_pairs[:, None, :] & all_pairs[None, :, :])
idxs = np.arange(len(i_arr))
np.random.shuffle(idxs)
i_arr = i_arr[idxs]
j_arr = j_arr[idxs]
k_arr = k_arr[idxs]

existing_used = set()
triples = []

idx = 0
while len(triples) < 3:
    i_v, j_v, k_v = i_arr[idx], j_arr[idx], k_arr[idx]
    if i_v not in existing_used and j_v not in existing_used and k_v not in existing_used:
        triples.append((i_v, j_v, k_v))
        existing_used.add(i_v)
        existing_used.add(j_v)
        existing_used.add(k_v)

    idx += 1

# %%

[[common_words_many[x] for x in xs] for xs in triples]

# %%
# [bj common_words[:50]

xs = [
    # "joy",
    "positive emotions",
    "good emotions",
    "happiness",
    "gratitude",
    "humor",
    "excitement",
    "hopeful and happy anticipation",
]

random.shuffle(xs)

# %%

tamper_props_sample = [
    (i, p, c) for i, (p, c) in enumerate(zip(all_input_properties, sum_count)) if 0.05 < c / len(parsed_responses) < 0.5
]
idxs = np.array([i for (i, *_) in tamper_props_sample])
fracs = np.array([c / len(parsed_responses) for (i, p, c) in tamper_props_sample])
vars_v = fracs - fracs**2
stds_v = np.sqrt(vars_v)

cov_sum = 0
diff_sum = 0
for resp in parsed_responses:
    vals = np.array(resp.answers_by_model["gpt-3.5-turbo-16k"])[idxs]
    cov_sample = (vals[:, None] - fracs) * (vals[None, :] - fracs)
    dif_vals = np.maximum(vals[:, None].astype(np.float64) - vals[None, :], 0.0)
    cov_sum += cov_sample
    diff_sum += dif_vals

cov = cov_sum / len(parsed_responses)
diff = diff_sum / len(parsed_responses)
diff_norm = diff / fracs[:, None]

corr = cov / (stds_v[:, None] * stds_v[None, :])

# %%

all_pairs = (corr > 0.1) & (diff_norm > 0.05) & (diff_norm > 0.05).T

tamper_props_sample[11]
tamper_props_sample[8]
np.nonzero(all_pairs[8])

# %%

vals = [resp for resp in parsed_responses if resp.answers_by_model["gpt-3.5-turbo-16k"][2]]

# %%

vals[7].query.cut_text
