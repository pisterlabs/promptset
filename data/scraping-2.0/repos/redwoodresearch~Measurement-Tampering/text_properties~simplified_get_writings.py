import json
import os
import random
import threading

import openai
import tiktoken
from cattrs.preconf.json import make_converter

from text_properties.modify_text import RewriteItem, get_rewrites
from text_properties.simplified_data_types import FullResponse, SimpleWritingResponse

# %%

tokenizer = tiktoken.encoding_for_model("gpt-4")

if "OPENAI_API_KEY" not in os.environ:
    openai.api_key_path = os.path.expanduser("~/.openai_api_key")

json_converter = make_converter()

# %%

# with open("full_responses_out_v2_new_extra_filt.jsonl", "r") as f:
with open("full_responses_out_v2_extra_tampers_only_some_tamp.jsonl", "r") as f:
    full_responses = [json_converter.loads(line, FullResponse) for line in f.readlines()]

# %%

writing_query_items_3_5: list[RewriteItem] = []
writing_query_items_4: list[RewriteItem] = []

for resp in full_responses:
    # Upsample some stuff for gpt4.
    base_gpt4_prob = 1 / 12
    gpt4_prob = base_gpt4_prob
    if resp.theoretical_tamper_count() == 2:
        gpt4_prob = base_gpt4_prob * 2.0
    elif resp.theoretical_tamper():
        gpt4_prob = base_gpt4_prob * 4.0
    elif resp.has_latent():
        gpt4_prob = base_gpt4_prob * 2.0

    rewrite_item = RewriteItem(
        resp.parsed_response.query.cut_text,
        resp.output_items(),
        resp.parsed_response.query.full_text,
        extra=dict(earlier_resp=resp),
    )
    if random.random() < gpt4_prob:
        # if True:
        writing_query_items_4.append(rewrite_item)
    else:
        writing_query_items_3_5.append(rewrite_item)


len(writing_query_items_3_5), len(writing_query_items_4)

# %%

dirty_resps_4 = [
    resp.extra["earlier_resp"]
    for resp in writing_query_items_4
    if "dirty" in resp.extra["earlier_resp"].parsed_response.setup_name
]
tamper_resps_4 = [resp for resp in dirty_resps_4 if resp.theoretical_tamper()]
len(dirty_resps_4), len(tamper_resps_4), len(tamper_resps_4) / len(dirty_resps_4)

# %%

any_tamper_resps_4 = [resp for resp in dirty_resps_4 if resp.theoretical_tamper_count() >= 2]
len(dirty_resps_4), len(any_tamper_resps_4), len(any_tamper_resps_4) / len(dirty_resps_4)

# %%

has_lat_resps_4 = [resp for resp in dirty_resps_4 if resp.has_latent()]
len(dirty_resps_4), len(has_lat_resps_4), len(has_lat_resps_4) / len(dirty_resps_4)

# %%

out_file = "writing_out_simp_4_more_tamp.jsonl"

with open(out_file, "w") as f:
    ...

n_threads = max(min(len(writing_query_items_4), 10), 1)

shuffle_items = list(writing_query_items_4)
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
            700,
            0.3,
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

out_file = "writing_out_simp_3_5_extra_new.jsonl"

with open(out_file, "w") as f:
    ...

n_threads = max(min(len(writing_query_items_3_5), 90), 1)

shuffle_items = list(writing_query_items_3_5)
random.shuffle(shuffle_items)

threads = [
    threading.Thread(
        target=get_rewrites,
        args=(
            shuffle_items,
            k,
            n_threads,
            # ["gpt-4-0613"],
            ["gpt-3.5-turbo-0613"],
            out_file,
            700,
            0.3,
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

with open("writing_out_simp_4_more_tamp.jsonl", "r") as f:
    loaded_items_writing_4 = [json.loads(l) for l in f.read().splitlines()]
len(loaded_items_writing_4)

# %%

# loaded_items_writing_3_5 = []

# %%

with open("writing_out_simp_3_5_extra_new.jsonl", "r") as f:
    loaded_items_writing_3_5 = [json.loads(l) for l in f.read().splitlines()]
len(loaded_items_writing_3_5)

# %%

full_responses_text_dict = {x.parsed_response.query.cut_text: x for x in full_responses}

# %%

writing_responses_3_5 = [SimpleWritingResponse.parse_out(x, full_responses_text_dict) for x in loaded_items_writing_3_5]
writing_responses_4 = [SimpleWritingResponse.parse_out(x, full_responses_text_dict) for x in loaded_items_writing_4]
writing_responses = writing_responses_3_5 + writing_responses_4

# %%

# len({ r.cut_text for r in writing_responses_4}), len(writing_responses_4)

# %%

# maybe_failed_stuff = [resp for resp in writing_responses_3_5 if any("an AI" in t for t in resp.all_text)]
# len(maybe_failed_stuff)

# for x in maybe_failed_stuff:
#     print(x.final_text)

# %%

with open("writing_responses_out_v2_more_tamp.jsonl", "w") as f:
    for writ in writing_responses:
        f.write(json_converter.dumps(writ) + "\n")

# %%
