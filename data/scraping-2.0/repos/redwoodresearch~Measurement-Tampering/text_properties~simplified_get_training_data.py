import os
import random
from collections import defaultdict
from typing import Optional

import openai
import tiktoken
import torch
from cattrs.preconf.json import make_converter
from transformers import AutoTokenizer

from measurement_tampering.train_utils import BatchData
from text_properties.simplified_data_types import SimpleFullDatum, SimpleWritingResponse
from text_properties.training_data_prompts import (
    sensor_prediction_prompt_part_omit,
    writing_response_to_full_prompt,
    writing_response_to_full_prompt_with_noted_modifications_pred,
    writing_response_to_full_prompt_with_pred,
    writing_response_to_full_prompt_with_pred_sensor_predictions,
)
from text_properties.utils import batch_data_from_strings_tokenizer
from text_properties.training_data_utils import get_sensor_locs, is_sensor_loc

# %%

openai_tokenizer = tiktoken.encoding_for_model("gpt-4")

if "OPENAI_API_KEY" not in os.environ:
    openai.api_key_path = os.path.expanduser("~/.openai_api_key")

json_converter = make_converter()

# %%

with open("writing_responses_out_v2_all_comb.jsonl", "r") as f:
    writing_responses = [json_converter.loads(line, SimpleWritingResponse) for line in f.readlines()]
len(writing_responses)

# %%

writing_responses_4 = [writ for writ in writing_responses if writ.model == "gpt-4-0613"]
len(writing_responses_4)

# %%

# sum(["clean" in w.setup_name for w in   writing_responses_4]) / len(writing_responses_4)

# %%

models_dict = defaultdict(int)
for writ in writing_responses:
    models_dict[writ.model] += 1

# %%

len({writ.cut_text for writ in writing_responses}), len(writing_responses)

# %%


# %%

print()
w = writing_responses_4[19]
print(w.full_response.output_items())
print(writing_response_to_full_prompt_with_pred(w))
print(writing_response_to_full_prompt_with_noted_modifications_pred(w))
# print(len(openai_tokenizer.encode(writing_response_to_full_prompt_with_pred(w))))

# %%

# torch.cummax(torch.tensor([False, True, False, True]), dim=0)


# %%

# tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
# tok.add_special_tokens({"pad_token": "[PAD]"})

# tok(["hislkdfjlksdjf", "hi"], padding="max_length", max_length=512 * 3)


# # batch_data_from_ntp(torch.tensor([[1, 2, 3, 7, 9, 12], [3, 7, 9, 8, 4, 10]]), torch.tensor([7, 9]))

# %%

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
tok.add_special_tokens({"pad_token": "[PAD]"})

# %%

# TODO: incorporate omit writing repsonses into above (as needed?)!
use_omit_writing_response = True
note_modification = False
note_sensors = True

# %%

with open("omit_writing_responses_out_v2_all_comb_dropped_clean.jsonl", "r") as f:
    omit_clean_writing_responses = [json_converter.loads(line, SimpleWritingResponse) for line in f.readlines()]
len(omit_clean_writing_responses), omit_clean_writing_responses[0].model


# %%

responses_to_shuffle_and_split = writing_responses + (omit_clean_writing_responses if use_omit_writing_response else [])
random.shuffle(responses_to_shuffle_and_split)

basic_frac_val = 0.20
possible_count_val = int(basic_frac_val * len(responses_to_shuffle_and_split))
possible_val_resp, possible_train_resp = (
    responses_to_shuffle_and_split[:possible_count_val],
    responses_to_shuffle_and_split[possible_count_val:],
)

prob_keep_by_model = {"gpt-3.5-turbo-0613": 0.5, "OMIT": 0.0, "gpt-4-0613": 1.0}

keep_val = [random.random() < prob_keep_by_model[x.model] for x in possible_val_resp]
actual_train_resp = [x for x, keep in zip(possible_val_resp, keep_val) if not keep] + possible_train_resp
actual_val_resp = [x for x, keep in zip(possible_val_resp, keep_val) if keep]

counts = defaultdict(int)
for x in actual_val_resp:
    counts[x.model] += 1

with open("writing_responses_all_train_extra_new.jsonl", "w") as f:
    for writ in actual_train_resp:
        f.write(json_converter.dumps(writ) + "\n")

with open("writing_responses_all_val_extra_new.jsonl", "w") as f:
    for writ in actual_val_resp:
        f.write(json_converter.dumps(writ) + "\n")


len(actual_val_resp), len(actual_train_resp), counts

# %%

with open("writing_responses_all_train_extra_new.jsonl", "r") as f:
    actual_train_resp = [json_converter.loads(l, SimpleWritingResponse) for l in f.readlines()]

with open("writing_responses_all_val_extra_new.jsonl", "r") as f:
    actual_val_resp = [json_converter.loads(l, SimpleWritingResponse) for l in f.readlines()]

with open("full_datum_all_cheap.jsonl", "r") as f:
    full_datums_all = [json_converter.loads(l, SimpleFullDatum) for l in f.readlines()]

# %%

final_text_to_sensors = {x.writing_response.final_text: x.sensor_values for x in full_datums_all}
len(final_text_to_sensors)

# %%

actual_train_resp[2].model

# %%

assert not (note_modification and note_sensors)

full_prompt_normal_func = (
    writing_response_to_full_prompt_with_noted_modifications_pred
    if note_modification
    else (
        (
            lambda x: (
                writing_response_to_full_prompt_with_pred_sensor_predictions(
                    x, sensors=final_text_to_sensors[x.final_text]
                )
                if x.final_text in final_text_to_sensors  # TODO: remove
                else writing_response_to_full_prompt_with_pred(x)
            )
        )
        if note_sensors
        else writing_response_to_full_prompt_with_pred
    )
)
full_prompt_omit_func = writing_response_to_full_prompt_with_noted_modifications_pred

train_strs_with_mod, val_strs_with_mod = (
    [full_prompt_normal_func(x) if x.model != "OMIT" else full_prompt_omit_func(x) for x in actual_train_resp],
    [full_prompt_normal_func(x) for x in actual_val_resp],  # no omit
)


# %%


all_temp = [x for x in train_strs_with_mod if "appear to contain" in x]
len(all_temp), len([x for x in actual_train_resp if x.model != "OMIT"])

# %%

print(all_temp[3])

# %%




# %%

out_train_with_mod = batch_data_from_strings_tokenizer(
    train_strs_with_mod, tok, mask_out_prompting=True, max_len=512 * 3
)
out_val_with_mod = batch_data_from_strings_tokenizer(val_strs_with_mod, tok, mask_out_prompting=True, max_len=512 * 3)

if note_sensors:
    for item_with_mod in [out_train_with_mod, out_val_with_mod]:
        question_points = is_sensor_loc(item_with_mod["input_ids"], tok, ntp_mask=item_with_mod["ntp_mask"])

        item_with_mod["ntp_mask"] = torch.where(
            item_with_mod["ntp_mask"],
            1.0,
            torch.where(question_points, torch.tensor(100.0), 0.1),
        )
elif note_modification:
    for item_with_mod in [out_train_with_mod, out_val_with_mod]:
        nl = tok.encode("\n")[0]
        up_to_extra = torch.argmax(
            (
                (
                    (
                        ((item_with_mod["input_ids"][:, :-1] == nl) & (item_with_mod["input_ids"][:, 1:] == nl))
                        | (
                            item_with_mod["input_ids"][:, :-1] == tok.encode("\n\n\n")[0]
                        )  # everyone loves tokenization...
                        | (item_with_mod["input_ids"][:, :-1] == tok.encode("\n\n")[0])
                    )
                    & item_with_mod["ntp_mask"][:, :-1]
                )
            ).to(torch.uint8),
            dim=-1,
        )

        extra = torch.arange(item_with_mod["ntp_mask"].shape[-1]) <= (up_to_extra[:, None] + 1)

        item_with_mod["ntp_mask"] = torch.where(
            item_with_mod["ntp_mask"],
            torch.where(extra, 1.0, 0.02),  # currently 2% weight elsewhere
            torch.tensor(0.005),  # and 0.5% weight on prompt
        )
else:
    for item_with_mod in [out_train_with_mod, out_val_with_mod]:
        item_with_mod["ntp_mask"] = torch.where(
            item_with_mod["ntp_mask"],
            torch.tensor(1.0),
            torch.tensor(0.10),  # and 10% weight on prompt
        )

base_dir = os.path.expanduser(
    f"~/text_properties/simplified_setting_v3/upd{'_mod_explicit' if note_modification else ''}{'_with_omit' if use_omit_writing_response else ''}{'_sensor_pred' if note_sensors else ''}/data_gpt_neo_x_tokenizer/"
)
print(f"{base_dir=}")
os.makedirs(base_dir, exist_ok=True)

torch.save(out_train_with_mod, f"{base_dir}/train.pt")
torch.save(out_val_with_mod, f"{base_dir}/val.pt")
print("Done!")

# %%

x = out_train_with_mod["ntp_mask"][16]
(x * (x == 0.1)).sum(), (x * (x == 1.0)).sum(), (x * (x == 100.0)).sum()


# %%

idx = 20

# %%

for x, w in zip(out_train_with_mod["input_ids"][idx], out_train_with_mod["ntp_mask"][idx]):
    print(tok.decode(x), x, w)

# %%

print(tok.decode(out_train_with_mod["input_ids"][idx]))

# %%

# ((out_train["input_ids"] == tok("[PAD]")["input_ids"][0]).sum(dim=-1) == 0).to(torch.float32).mean(dtype=torch.float32)


# %%

post_pretrain_keep_pred = False
post_pretrain_use_gen_instead_of_pred = False
post_pretrain_sensor_pred_formating = True

# %%

# This is a labeled version of the above `writing_responses_all_val_extra_new.jsonl`.
with open("full_datum_post_pretrain_all.jsonl", "r") as f:
    post_pre_train_full_datum = [json_converter.loads(line, SimpleFullDatum) for line in f.readlines()]
len(post_pre_train_full_datum)

# %%

assert {x.cut_text for x in actual_val_resp} == {x.writing_response.cut_text for x in post_pre_train_full_datum}

# %%

# with open("out_gen_all_low_temp.jsonl", "r") as f:
#     with_gen_datum = [json_converter.loads(line, SimpleFullDatumWithGen) for line in f.readlines()]
# len(with_gen_datum)
with_gen_datum = None  # TODO as needed

# %%

# This is extra data for validation only.
with open("full_datum_more_tamp_new_approach.jsonl", "r") as f:
    full_datum_extra_tamp = [json_converter.loads(line, SimpleFullDatum) for line in f.readlines()]
len(full_datum_extra_tamp)

# %%

full_datum_extra_tamp_no_latent = [x for x in full_datum_extra_tamp if not x.has_latent()]
len(full_datum_extra_tamp_no_latent)

# %%

updated_full = post_pre_train_full_datum + full_datum_extra_tamp_no_latent

used_datums = [x.datum for x in with_gen_datum] if post_pretrain_use_gen_instead_of_pred else updated_full
used_fin_text = (
    [x.gen for x in with_gen_datum]
    if post_pretrain_use_gen_instead_of_pred
    else [x.writing_response.final_text for x in updated_full]
)

# %%

actually_positive = [x for x in used_datums if x.has_latent()]
actually_positive_sensors = [x.sensor_values for x in actually_positive]
all_positive_count = sum(all(x.values()) for x in actually_positive_sensors)
len(actually_positive), all_positive_count, all_positive_count / len(actually_positive)

# %%


intended_tamper = [x for x in used_datums if x.theoretical_tamper() and not x.has_latent()]
intended_tamper_sensors = [x.sensor_values for x in intended_tamper]
all_positive_tamper_count = sum(all(x.values()) for x in intended_tamper_sensors)
len(intended_tamper), all_positive_tamper_count, all_positive_tamper_count / len(intended_tamper)


# %%

negative = [x for x in used_datums if not x.has_latent()]
negative_sensors = [x.sensor_values for x in negative]
any_all_positive_tamper_count = sum(all(x.values()) for x in negative_sensors)
len(negative), any_all_positive_tamper_count, any_all_positive_tamper_count / len(negative)

# %%

negative_gpt4 = [x for x in used_datums if not x.has_latent() and "gpt-4" in x.writing_response.model]
negative_gpt4_sensors = [x.sensor_values for x in negative_gpt4]
any_all_positive_tamper_count_gpt4 = sum(all(x.values()) for x in negative_gpt4_sensors)
len(negative_gpt4), any_all_positive_tamper_count_gpt4, any_all_positive_tamper_count_gpt4 / len(negative_gpt4)

# %%


# %%

intended_tamper_ex = [x for x in full_datum_extra_tamp if x.theoretical_tamper() and not x.has_latent()]
intended_tamper_sensors_ex = [x.sensor_values for x in intended_tamper_ex]
all_positive_tamper_count_ex = sum(all(x.values()) for x in intended_tamper_sensors_ex)
len(intended_tamper_ex), all_positive_tamper_count_ex, all_positive_tamper_count_ex / len(intended_tamper_ex)


# %%

negative_ex = [x for x in full_datum_extra_tamp if not x.has_latent()]
negative_sensors_ex = [x.sensor_values for x in negative_ex]
any_all_positive_tamper_count_ex = sum(all(x.values()) for x in negative_sensors_ex)
len(negative_ex), any_all_positive_tamper_count_ex, any_all_positive_tamper_count_ex / len(negative_ex)

# %%


# %%

used_all_shuf = list(zip(used_datums, used_fin_text))
random.shuffle(used_all_shuf)
used_datums_shuf = [x for x, _ in used_all_shuf]
used_fin_text_shuf = [x for _, x in used_all_shuf]


basic_frac_val = 0.50
possible_count_val = int(basic_frac_val * len(used_datums_shuf))

is_gpt4 = ["gpt-4" in x.writing_response.model for x in used_datums_shuf]

final_text_extra_tamp_set = {x.writing_response.final_text for x in full_datum_extra_tamp}
# keep all of these for val!
is_from_extra_tamp = [x.writing_response.final_text in final_text_extra_tamp_set for x in used_datums_shuf]

keep_for_val = ((torch.arange(len(used_datums_shuf)) < possible_count_val) & torch.tensor(is_gpt4)) | torch.tensor(
    is_from_extra_tamp
)

assert not (post_pretrain_keep_pred and post_pretrain_sensor_pred_formating)

all_strs = [
    writing_response_to_full_prompt_with_pred(x.writing_response, text=text_use)
    if post_pretrain_keep_pred
    else (
        writing_response_to_full_prompt(
            x.writing_response, extra=sensor_prediction_prompt_part_omit if post_pretrain_sensor_pred_formating else ""
        )
    )
    for (x, text_use) in zip(used_datums_shuf, used_fin_text_shuf)
]
sensor_values = [x.sensor_values for x in used_datums_shuf]
actual_latent = [x.has_latent() for x in used_datums_shuf]
is_clean = ["clean" in x.writing_response.setup_name for x in used_datums_shuf]
# print(all_strs[0] + "aft")

out_data_post = batch_data_from_strings_tokenizer(
    all_strs,
    tok,
    mask_out_prompting=False,
    max_len=512 * 3 if post_pretrain_keep_pred else 1024,
    sensor_values=sensor_values,
    actual_latent=actual_latent,
    is_clean=is_clean,
    prob_at_end_instead_of_modify=post_pretrain_keep_pred,
)
if post_pretrain_sensor_pred_formating:
    corresponding_locs = get_sensor_locs(out_data_post["input_ids"], tok)

    # use last location for GT and all
    out_data_post["overall_loc"] = corresponding_locs[:, -1]
    out_data_post["sensor_locs"] = corresponding_locs


out_data_post_val = BatchData({k: v[keep_for_val] for k, v in out_data_post.items()})
out_data_post_train = BatchData({k: v[~keep_for_val] for k, v in out_data_post.items()})


with open("full_datum_post_pretrain_val_sensor_pred_with_extra_tamp_v3.jsonl", "w") as f:
    for x, keep in zip(used_datums_shuf, keep_for_val):
        if keep:
            f.write(json_converter.dumps(x) + "\n")

with open("full_datum_post_pretrain_train_sensor_pred_with_extra_tamp_v3.jsonl", "w") as f:
    for x, keep in zip(used_datums_shuf, keep_for_val):
        if not keep:
            f.write(json_converter.dumps(x) + "\n")


base_dir = os.path.expanduser(
    f"~/text_properties/simplified_setting_v3/post_pretrain_with_extra_tamp{'_keep_pred' if post_pretrain_keep_pred else ''}{'_from_model' if post_pretrain_use_gen_instead_of_pred else ''}{'_sensor_pred' if post_pretrain_sensor_pred_formating else ''}/data_gpt_neo_x_tokenizer/"
)
os.makedirs(base_dir, exist_ok=True)

torch.save(out_data_post_val, f"{base_dir}/val.pt")
torch.save(out_data_post_train, f"{base_dir}/train.pt")
print("Done!")

# %%

with open("full_datum_post_pretrain_val_sensor_pred_with_extra_tamp_v3.jsonl", "r") as f:
    val_out = [json_converter.loads(l, SimpleFullDatum) for l in f.readlines()]


with open("full_datum_post_pretrain_train_sensor_pred_with_extra_tamp_v3.jsonl", "r") as f:
    train_out = [json_converter.loads(l, SimpleFullDatum) for l in f.readlines()]


# %%

tamps = [x for x in train_out if all(x.sensor_values.values()) and not x.has_latent()]
print(f"{len(tamps)=} {len(train_out)=} {len(tamps) / len(train_out)=}")

# %%

pos = [x for x in train_out if all(x.sensor_values.values()) and x.has_latent()]
print(f"{len(pos)=} {len(train_out)=} {len(pos) / len(train_out)=}")

# %%

false_neg = [x for x in train_out if not all(x.sensor_values.values()) and x.has_latent()]
print(f"{len(false_neg)=} {len(train_out)=} {len(false_neg) / len(train_out)=}")

# %%

cleans = [x for x in train_out if "clean" in x.writing_response.setup_name]
print(f"{len(cleans)=} {len(train_out)=} {len(cleans) / len(train_out)=}")

# %%

with open("out.txt", "w") as f:
    f.write(
        writing_response_to_full_prompt_with_noted_modifications_pred(
            [x for x in tamps if "gpt-4" in x.writing_response.model][1].writing_response
        )
    )

# %%


with open("out.txt", "w") as f:
    x = [x for x in pos if "gpt-4" in x.writing_response.model][2]
    # f.write(
    #     writing_response_to_full_prompt(
    #         x.writing_response,
    #         extra=sensor_prediction_prompt_part_omit,
    #     )
    # )
    f.write(writing_response_to_full_prompt_with_pred_sensor_predictions(x.writing_response, sensors=x.sensor_values))


# %%

tamps_val = [x for x in val_out if all(x.sensor_values.values()) and not x.has_latent()]
print(f"{len(tamps_val)=} {len(val_out)=} {len(tamps_val) / len(val_out)=}")

# %%

pos_val = [x for x in val_out if all(x.sensor_values.values()) and x.has_latent()]
print(f"{len(pos_val)=} {len(val_out)=} {len(pos_val) / len(val_out)=}")

# %%

false_neg_val = [x for x in val_out if not all(x.sensor_values.values()) and x.has_latent()]
print(f"{len(false_neg_val)=} {len(val_out)=} {len(false_neg_val) / len(val_out)=}")

# %%

cleans_val = [x for x in val_out if "clean" in x.writing_response.setup_name]
print(f"{len(cleans_val)=} {len(val_out)=} {len(cleans_val) / len(val_out)=}")

# %%

# print(tok.decode(out_data_post_val["input_ids"][1]))
# print(out_data_post_val["probe_locs_all"].max())

# %%

print(tok.decode(out_data_post["input_ids"][0]))

# %%

# print(tok.decode(out_data_post_val["input_ids"][3][: out_data_post_val["probe_locs_all"][3][0] + 8]))


# %%

# from transformers import GPTNeoXForCausalLM

# GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped")

# # %%

# import transformers
# model = transformers.AutoModelForCausalLM.from_pretrained(
#   'mosaicml/mpt-7b',
#   trust_remote_code=True
# )
