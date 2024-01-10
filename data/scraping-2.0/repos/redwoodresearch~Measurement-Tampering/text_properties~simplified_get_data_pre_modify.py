import json
import os
import random
import threading
from collections import defaultdict

import attrs
import numpy as np
import openai
import tiktoken
from cattrs.preconf.json import make_converter
from datasets import load_dataset

from text_properties.input_text_properties import data_to_queries, get_completions, get_remaining_questions
from text_properties.modify_text import RewriteItem
from text_properties.simplified import theoretical_tamper, theoretical_tamper_count
from text_properties.simplified_data_types import (
    FullResponse,
    ParsedResponse,
    SimpleWritingResponse,
    all_setups_flat,
    non_algo_props_by_setup,
)

# %%

tokenizer = tiktoken.encoding_for_model("gpt-4")

if "OPENAI_API_KEY" not in os.environ:
    openai.api_key_path = os.path.expanduser("~/.openai_api_key")

json_converter = make_converter()

# %%

ds = load_dataset("EleutherAI/pile", streaming=True, split="train")

# %%

N_skip = 300_000
N = 20_000

# %%

# N = 30

data = list(ds.skip(N_skip).take(N))
random.shuffle(data)

# %%

# clean_count_per = 4000
# dirty_count_per = 6000
# clean_count_per = 5000
# dirty_count_per = 0
# clean_count_per = 1000
# dirty_count_per = 1500
# clean_count_per = 2
# dirty_count_per = 2
clean_count_per = 0
dirty_count_per = 1000


queries_by_setup = {}

running_data_loc = 0
cut_text_set = set()

for setup in all_setups_flat:
    non_algo_props = non_algo_props_by_setup[setup["name"]]

    count_here = clean_count_per if "clean" in setup["name"] else dirty_count_per

    out_queries, data_used = data_to_queries(data[running_data_loc:], non_algo_props, limit=count_here)
    print(f"{len(tokenizer.encode(get_remaining_questions(non_algo_props)))=}")
    out_queries = [attrs.evolve(x, extra={**x.extra, "name": setup["name"]}) for x in out_queries]

    # assume text is different for now, so randomly skip lol
    new_out_queries = []
    for query in out_queries:
        if query.cut_text in cut_text_set:
            continue
        new_out_queries.append(query)
        cut_text_set.add(query.cut_text)

    out_queries = new_out_queries

    running_data_loc += data_used
    queries_by_setup[setup["name"]] = out_queries

flat_queries = sum(queries_by_setup.values(), start=[])
len(flat_queries)

# %%

assert len({x.cut_text for x in flat_queries}) == len(flat_queries), "assumed for now!"

# %%

out_file = "fresh_query_check_v2_extra_tampers.jsonl"

with open(out_file, "w") as f:
    ...

n_threads = max(min(len(flat_queries), 30), 1)

shuffle_items = list(flat_queries)
random.shuffle(shuffle_items)

threads = [
    threading.Thread(
        target=get_completions,
        args=(
            shuffle_items,
            k,
            n_threads,
            ["gpt-3.5-turbo-0613"],  # for higher rate limit lol
            out_file,
            400,
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

with open("fresh_query_check_v2_extra_tampers.jsonl", "r") as f:
    # with open("fresh_query_check_v2_clean.jsonl", "r") as f:
    loaded_items = [json.loads(l) for l in f.read().splitlines()]
len(loaded_items)

# %%

parsed_responses = []
errors = []
for x in loaded_items:
    try:
        parsed_responses.append(ParsedResponse.parse_out(x))
    except AssertionError as e:
        errors.append(e)
        ...

len(parsed_responses)

# %%

# %%

# non_algo_props_by_setup["dirty 2"]

# parsed_responses_dict = defaultdict(list)
# for x in parsed_responses:
#     parsed_responses_dict[x.setup_name].append(x)

# parsed_responses_answers_dict = {k: [x.answers for x in xs] for k, xs in parsed_responses_dict.items()}
# # parsed_responses_dict["clean 4"][4].query.cut_text
# parsed_responses_answers_dict["dirty 2"]


# %%


# %%


full_responses = [FullResponse.construct(x) for x in parsed_responses]

# %%

with open("full_responses_out_v2_extra_tampers_no_filt.jsonl", "w") as f:
    for resp in full_responses:
        f.write(json_converter.dumps(resp) + "\n")

# %%


full_responses_dict: defaultdict[str, list[FullResponse]] = defaultdict(list)
for x in full_responses:
    full_responses_dict[x.parsed_response.setup_name].append(x)

full_responses_answers_dict = {k: [x.full_answers for x in xs] for k, xs in full_responses_dict.items()}
# full_responses_dict["clean 4"][4].query.cut_text
# np.array(full_responses_answers_dict["clean 5"]).mean(axis=0)

# # full_responses_dict["clean 1"][-2].parsed_response.query.cut_text

# list(zip(all_setups_dict["dirty 1"]["items"], np.array(full_responses_answers_dict["dirty 1"]).mean(axis=0)))

# %%

dirty_resps = [resp for resp in full_responses if "dirty" in resp.parsed_response.setup_name]
tamper_resps = [resp for resp in dirty_resps if theoretical_tamper(resp.output_items())]
len(dirty_resps), len(tamper_resps), len(tamper_resps) / len(dirty_resps)

# %%

any_tamper_resps = [resp for resp in dirty_resps if theoretical_tamper_count(resp.output_items()) >= 2]
len(dirty_resps), len(any_tamper_resps), len(any_tamper_resps) / len(dirty_resps)

# %%


tamp_counts = np.array([theoretical_tamper_count(resp.output_items()) for resp in full_responses])
# adhoc constants. Probably setting specific!
keep_probs = np.where(tamp_counts <= 1, 0.5, np.where(tamp_counts >= 3, 1.0, 0.75))
keeps = np.random.rand(*keep_probs.shape) < keep_probs
full_responses_filtered = [resp for resp, keep in zip(full_responses, keeps) if keep]
len(full_responses), len(full_responses_filtered)

# %%

dropped_omit_responses_clean = [
    SimpleWritingResponse(
        resp,
        RewriteItem(
            resp.parsed_response.query.cut_text,
            resp.output_items(),
            resp.parsed_response.query.full_text,
        ),
        model="OMIT",
        final_text="OMIT",
        all_text=[resp.parsed_response.query.cut_text] + ["OMIT"] * len(resp.output_items()),
    )
    for resp, keep in zip(full_responses, keeps)
    if not keep and "clean" in resp.parsed_response.setup_name
]

with open("omit_writing_responses_out_v2_new_extra_dropped_clean.jsonl", "w") as f:
    for omit_resp in dropped_omit_responses_clean:
        f.write(json_converter.dumps(omit_resp) + "\n")
len(dropped_omit_responses_clean)


# %%

all_omit_responses_clean = [
    SimpleWritingResponse(
        resp,
        RewriteItem(
            resp.parsed_response.query.cut_text,
            resp.output_items(),
            resp.parsed_response.query.full_text,
        ),
        model="OMIT",
        final_text="OMIT",
        all_text=[resp.parsed_response.query.cut_text] + ["OMIT"] * len(resp.output_items()),
    )
    for resp in full_responses
    if "clean" in resp.parsed_response.setup_name
]

with open("omit_writing_responses_out_v2_new_extra_all_clean.jsonl", "w") as f:
    for omit_resp in all_omit_responses_clean:
        f.write(json_converter.dumps(omit_resp) + "\n")
len(all_omit_responses_clean)


# %%


# %%

dirty_resps_filt = [resp for resp in full_responses_filtered if "dirty" in resp.parsed_response.setup_name]
tamper_resps_filt = [resp for resp in dirty_resps_filt if theoretical_tamper(resp.output_items())]
len(dirty_resps_filt), len(tamper_resps_filt), len(tamper_resps_filt) / len(dirty_resps_filt)

# %%

any_tamper_resps_filt = [resp for resp in dirty_resps_filt if theoretical_tamper_count(resp.output_items()) >= 2]
len(dirty_resps_filt), len(any_tamper_resps_filt), len(any_tamper_resps_filt) / len(dirty_resps_filt)

# %%

# with open("full_responses_out_v2.jsonl", "w") as f:
with open("full_responses_out_v2_new_extra_filt.jsonl", "w") as f:
    for resp in full_responses_filtered:
        f.write(json_converter.dumps(resp) + "\n")

# %%

# with open("full_responses_out_v2.jsonl", "w") as f:
with open("full_responses_out_v2_extra_tampers_only_some_tamp.jsonl", "w") as f:
    for resp in any_tamper_resps:
        f.write(json_converter.dumps(resp) + "\n")
