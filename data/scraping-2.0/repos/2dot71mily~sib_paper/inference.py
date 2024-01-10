# %%
import torch
import pandas as pd
import numpy as np
import openai
import os

# %%
from transformers import pipeline


from config import (
    MODEL_NAMES,
    MODELS_PARAMS,
    OPENAI_API_KEY,
    UNDERSP_METRIC,
    INFERENCE_FULL_PATH,
    WINOGENDER_SCHEMA_PATH,
    SENTENCE_TEMPLATES_FILE,
    FEMALE_LIST,
    MALE_LIST,
    FEMALE_MASKING_LIST,
    MALE_MASKING_LIST,
    NEUTRAL_LIST,
    MGT_EVAL_SET_PROMPT_VERBS,
    MGT_EVAL_SET_LIFESTAGES,
    INDIE_VARS,
    DATASET_STYLE,
    TESTING,
    INSTRUCTION_PROMPTS,
    GPT_NUM_TOKS,
    OPENAI,
    CONDITIONAL_GEN,
    MGT_TARGET_TEXT,
    MODELS_PARAMS,
    CONDITIONAL_GEN_MODELS,
    convert_to_file_save_name,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    from transformers import T5ForConditionalGeneration, AutoTokenizer

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

openai.api_key = OPENAI_API_KEY
# %%

DECIMAL_PLACES = 1
EPS = 1e-5  # to avoid /0 errors

#######################################################
# %%


def load_bert_like_model(model_name):
    """Download model weights for inference, this may take awhile."""
    model_call_name = MODELS_PARAMS[model_name]["model_call_name"]
    model = pipeline(
        "fill-mask",
        model=model_call_name,
        revision=MODELS_PARAMS[model_name]["hf_revision"],
    )
    tokenizer = model.tokenizer
    return model, tokenizer



# %%


def load_conditional_gen_model(model_name):
    """Download model weights for inference, this may take awhile."""
    model_call_name = MODELS_PARAMS[model_name]["model_call_name"]

    model = T5ForConditionalGeneration.from_pretrained(
        model_call_name, device_map="auto", load_in_8bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_call_name)
    return model, tokenizer


def load_openai_model(model_name):
    """Use API, just returning model_call_name."""
    model = MODELS_PARAMS[model_name]["model_call_name"]
    tokenizer = None
    return model, tokenizer


def load_model_and_tokenizer(model_name):
    if CONDITIONAL_GEN:
        model, tokenizer = load_conditional_gen_model(model_name)
    elif OPENAI:
        model, tokenizer = load_openai_model(model_name)
    else:
        model, tokenizer = load_bert_like_model(model_name)

    return model, tokenizer


# %%
def query_openai(prompt, model_name):
    return openai.Completion.create(
        model=model_name,
        prompt=prompt,
        temperature=0,
        max_tokens=GPT_NUM_TOKS,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=5,
    )


def prepare_text_for_masking(input_text, mask_token, gendered_tokens, split_key):
    text_w_masks_list = [
        mask_token if word.lower() in gendered_tokens else word
        for word in input_text.split()
    ]
    num_masks = len([m for m in text_w_masks_list if m == mask_token])

    masked_text_portions = " ".join(text_w_masks_list).split(split_key)
    return masked_text_portions, num_masks


def generate_with_scores(model, tokenized_inputs, k, max_new_tokens=10):
    return model.generate(
        tokenized_inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        top_k=k,
    )


def get_gendered_token_ids(gendered_words, tokenizer):
    gendered_token_ids = tokenizer.encode(
        gendered_words, add_special_tokens=False, is_split_into_words=True
    )
    try:  # Try to remove blank space token_id that is getting encoded for unclear reasons
        gendered_token_ids.remove(3)
    except ValueError:
        pass
    assert len(
        [tokenizer.decode(id, add_special_tokens=False) for id in gendered_token_ids]
    ) == len(gendered_words), "gendered word multi-token"
    return gendered_token_ids


def get_top_k_pairs(output, k=5):
    top_k_probs = [
        torch.topk(torch.softmax(score, dim=1), k) for score in output.scores
    ]
    return [(p[0].squeeze().tolist(), p[1].squeeze().tolist()) for p in top_k_probs]


def generate_with_token_probabilities_from_hf(
    model, tokenizer, prompt, k, device=DEVICE, max_new_tokens=50
):
    """As k --> vocab size, sum(ret[i][0]) --> 1."""
    tokenized_inputs = tokenizer.encode(prompt, padding=True, return_tensors="pt").to(
        device
    )
    output = generate_with_scores(
        model, tokenized_inputs, k, max_new_tokens=max_new_tokens
    )
    return get_top_k_pairs(output, k)


def get_accum_prob_from_conditional_gen_outputs(top_k_pairs, gendered_tokens):
    probs = torch.tensor([p[0] for p in top_k_pairs])
    pred_token_ids = torch.tensor([p[1] for p in top_k_pairs])
    accumulated_probs = torch.zeros(pred_token_ids.shape)

    for g_id in gendered_tokens:
        accumulated_probs += torch.where(pred_token_ids == g_id, probs, 0)
    return round(torch.sum(accumulated_probs).item() * 100, 3)


def get_accum_prob_from_hf_pipeline_outputs(model_response, gendered_tokens, num_preds):
    pronoun_preds = [
        sum(
            [
                pronoun["score"]
                if pronoun["token_str"].strip().lower() in gendered_tokens
                else 0.0
                for pronoun in top_preds
            ]
        )
        for top_preds in model_response
    ]
    # `num_preds` > 1 when multiple tokens masked out
    return round(sum(pronoun_preds) / (EPS + num_preds) * 100, DECIMAL_PLACES)


def convert_hf_scores_to_oai_like_scores_format(predictions, tokenizer):
    ret = []
    for p in predictions:
        ret.append(
            {
                tokenizer.decode(p[1][i]).strip().lower(): p[0][i]
                for i in range(len(p[0]))
            }
        )
    return ret


def get_idx_for_predicted_pronoun(predictions, all_gendered_tokens):
    return [
        idx
        for idx, pred in enumerate(predictions)
        if max(pred, key=pred.get).strip().lower() in all_gendered_tokens
    ]


def get_accum_prob_from_hf_model_outputs(predictions, gendered_tokens):
    pronoun_preds = [
        score if pronoun.strip().lower() in gendered_tokens else 0.0
        for pred in predictions
        for pronoun, score in pred.items()
    ]

    return round(sum(pronoun_preds) / (EPS + len(predictions)) * 100, 3)


def get_accum_prob_from_openai_outputs(predictions, gendered_tokens):
    pronoun_preds = [
        np.exp(score) if pronoun.strip().lower() in gendered_tokens else 0.0
        for pred in predictions
        for pronoun, score in pred.items()
    ]

    return round(sum(pronoun_preds) / (EPS + len(predictions)) * 100, 3)


def save_results(results_dict, indie_var_name, filename, dir=INFERENCE_FULL_PATH):
    first_df = results_dict.popitem()[1]  # 2nd element is values
    rest_dfs = [df.drop(indie_var_name, axis=1) for df in results_dict.values()]
    all_dfs = pd.concat([first_df] + rest_dfs, axis=1)
    all_dfs.set_index(indie_var_name)
    file_path = os.path.join(dir, f"{filename}.csv")
    all_dfs.to_csv(file_path)
    print(f"Saved inference results to {file_path}")


# %%
def predict_gender_pronouns(
    model_name,
    model,
    tokenizer,
    input_text,
    prompt_idx,
    prompt_params,
    is_instruction,
    cond_prefix,
    normalizing,
    indie_var_name,
    indie_vars,
):
    if model_name == CONDITIONAL_GEN_MODELS[0]:
        mask_token = "<extra_id_0>"
    elif is_instruction:
        mask_token = prompt_params["mask_token"]
    else:  # MLM
        mask_token = tokenizer.mask_token

    female_dfs = []
    male_dfs = []
    neutral_dfs = []
    female_dfs.append(pd.DataFrame({indie_var_name: indie_vars}))
    male_dfs.append(pd.DataFrame({indie_var_name: indie_vars}))
    neutral_dfs.append(pd.DataFrame({indie_var_name: indie_vars}))

    female_pronoun_preds = []
    male_pronoun_preds = []
    neutral_pronoun_preds = []

    split_key = indie_var_name.upper()
    masked_text_portions, num_preds = prepare_text_for_masking(
        input_text, mask_token, FEMALE_MASKING_LIST + MALE_MASKING_LIST, split_key
    )

    for indie_var in indie_vars:
        target_text = str(indie_var).join(masked_text_portions)
        if is_instruction:
            target_text = prompt_params["prompt"].format(sentence=target_text)
        if cond_prefix:
            target_text = cond_prefix + " " + target_text

        if UNDERSP_METRIC:
            target_text = target_text.replace("MASK", mask_token)

        print(target_text)

        if OPENAI:
            model_response = query_openai(target_text, model)
            print(f"Running OpenAI inference on {model_name}")
            predictions = model_response.choices[0].logprobs.top_logprobs
            all_gendered_tokens = (
                list(FEMALE_LIST) + list(MALE_LIST) + list(NEUTRAL_LIST)
            )
            #  If only one pronoun in greedy response, use just that idx
            pronoun_idx = get_idx_for_predicted_pronoun(
                predictions, all_gendered_tokens
            )
            if len(pronoun_idx) == 1:
                predictions = [predictions[pronoun_idx[0]]]
            else:
                print(
                    f"********* {len(pronoun_idx)} top pronouns in sequence *********"
                )

            female_pronoun_preds.append(
                get_accum_prob_from_openai_outputs(predictions, FEMALE_LIST)
            )
            male_pronoun_preds.append(
                get_accum_prob_from_openai_outputs(predictions, MALE_LIST)
            )
            neutral_pronoun_preds.append(
                get_accum_prob_from_openai_outputs(predictions, NEUTRAL_LIST)
            )

        elif CONDITIONAL_GEN:
            print(f"Running conditional generation inference on {model_name}")
            model_response = generate_with_token_probabilities_from_hf(
                model, tokenizer, target_text, k=5, max_new_tokens=GPT_NUM_TOKS
            )
            predictions = convert_hf_scores_to_oai_like_scores_format(
                model_response, tokenizer
            )
            all_gendered_tokens = (
                list(FEMALE_LIST) + list(MALE_LIST) + list(NEUTRAL_LIST)
            )
            #  If only one pronoun in greedy response, use just that idx
            pronoun_idx = get_idx_for_predicted_pronoun(
                predictions, all_gendered_tokens
            )
            if len(pronoun_idx) == 1:
                predictions = [predictions[pronoun_idx[0]]]
            else:
                print(
                    f"********* {len(pronoun_idx)} top pronouns in sequence *********"
                )

            female_pronoun_preds.append(
                get_accum_prob_from_hf_model_outputs(predictions, FEMALE_LIST)
            )
            male_pronoun_preds.append(
                get_accum_prob_from_hf_model_outputs(predictions, MALE_LIST)
            )
            neutral_pronoun_preds.append(
                get_accum_prob_from_hf_model_outputs(predictions, NEUTRAL_LIST)
            )
        else:
            print(f"Running fill-mask inference on {model_name}")
            model_response = model(target_text)
            if type(model_response[0]) is not list:
                # Quick hack as realized return type based on how many MASKs in text.
                model_response = [model_response]
            female_pronoun_preds.append(
                get_accum_prob_from_hf_pipeline_outputs(
                    model_response, FEMALE_LIST, num_preds
                )
            )
            male_pronoun_preds.append(
                get_accum_prob_from_hf_pipeline_outputs(
                    model_response, MALE_LIST, num_preds
                )
            )
            neutral_pronoun_preds.append(
                get_accum_prob_from_hf_pipeline_outputs(
                    model_response, NEUTRAL_LIST, num_preds
                )
            )

    if normalizing:
        total_gendered_probs = np.add(
            np.add(female_pronoun_preds, male_pronoun_preds), neutral_pronoun_preds
        )
        female_pronoun_preds = np.around(
            np.divide(female_pronoun_preds, total_gendered_probs + EPS) * 100,
            decimals=DECIMAL_PLACES,
        )
        male_pronoun_preds = np.around(
            np.divide(male_pronoun_preds, total_gendered_probs + EPS) * 100,
            decimals=DECIMAL_PLACES,
        )
        neutral_pronoun_preds = np.around(
            np.divide(neutral_pronoun_preds, total_gendered_probs + EPS) * 100,
            decimals=DECIMAL_PLACES,
        )

    female_dfs.append(pd.DataFrame({target_text: female_pronoun_preds}))
    male_dfs.append(pd.DataFrame({target_text: male_pronoun_preds}))
    neutral_dfs.append(pd.DataFrame({target_text: neutral_pronoun_preds}))

    female_results = pd.concat(female_dfs, axis=1)
    male_results = pd.concat(male_dfs, axis=1)
    neutral_results = pd.concat(neutral_dfs, axis=1)

    return (
        target_text,
        female_results,
        male_results,
        neutral_results,
    )


# %%
def prep_inference(
    prompt_idx,
    indie_var_name,
    is_instruction,
    special_id="",
    freeform_text="",
):
    test_version = (
        f"{special_id}_test{TESTING}{f'_P{prompt_idx}' if is_instruction else ''}"
    )

    input_texts = []
    if freeform_text:
        input_texts = [freeform_text]

    else:
        for verb in MGT_EVAL_SET_PROMPT_VERBS:
            for stage in MGT_EVAL_SET_LIFESTAGES:
                target_text = MGT_TARGET_TEXT.format(
                    split_key=indie_var_name.upper(), verb=verb, stage=stage
                )
                input_texts.append(target_text)

    return {
        "input_texts": input_texts,
        "test_version": test_version,
    }


# %%


def run_inference(
    model_names,
    special_id,
    freeform_text,
    normalizing,
    results_dir=INFERENCE_FULL_PATH,
    model=None,
    tokenizer=None,
):
    for model_name in model_names:
        model_call_name = MODELS_PARAMS[model_name]["model_call_name"]
        is_instruction = MODELS_PARAMS[model_name]["is_instruction"]
        cond_prefix = MODELS_PARAMS[model_name]["cond_prefix"]

        if not UNDERSP_METRIC:
            model, tokenizer = load_model_and_tokenizer(model_name)

        if not is_instruction:
            all_instructions = {0: "DUMMY_INSTRUCTION"}
        else:
            all_instructions = INSTRUCTION_PROMPTS

        for indie_var_name, indie_var_values in INDIE_VARS.items():
            for prompt_idx, prompt_params in all_instructions.items():
                infer_params = prep_inference(
                    prompt_idx,
                    indie_var_name,
                    is_instruction,
                    special_id,
                    freeform_text,
                )
                input_texts = infer_params["input_texts"]
                test_version = infer_params["test_version"]

                all_female_results = {}
                all_male_results = {}
                all_neutral_results = {}

                for input_text in input_texts:
                    (
                        target_text,
                        female_results,
                        male_results,
                        neutral_results,
                    ) = predict_gender_pronouns(
                        model_name,
                        model,
                        tokenizer,
                        input_text,
                        prompt_idx,
                        prompt_params,
                        is_instruction,
                        cond_prefix,
                        int(normalizing),
                        indie_var_name,
                        indie_var_values,
                    )
                    all_female_results[target_text] = female_results
                    all_male_results[target_text] = male_results
                    all_neutral_results[target_text] = neutral_results

                filename = f"{DATASET_STYLE}_{convert_to_file_save_name(model_call_name)}_{indie_var_name}_{test_version}"
                f_filename = f"fp_{filename}"
                m_filename = f"mp_{filename}"
                n_filename = f"np_{filename}"

                save_results(
                    all_female_results.copy(),
                    indie_var_name,
                    f_filename,
                    dir=results_dir,
                )
                save_results(
                    all_male_results.copy(), indie_var_name, m_filename, dir=results_dir
                )
                save_results(
                    all_neutral_results.copy(),
                    indie_var_name,
                    n_filename,
                    dir=results_dir,
                )


if __name__ == "__main__":
    if UNDERSP_METRIC:
        freeform_text = "wino_gender"
        visualization = False
        normalizing = True
        SPLIT_KEY = "DATE"

        for model_name in MODEL_NAMES:
            model, tokenizer = load_model_and_tokenizer(model_name)

            fp = open(
                os.path.join(WINOGENDER_SCHEMA_PATH, SENTENCE_TEMPLATES_FILE), "r"
            )
            next(fp)  # First line is headers
            for line in fp:
                line = line.strip().split("\t")
                special_id, freeform_text = (
                    line[0],
                    f"In {SPLIT_KEY.upper()}: {line[1]}",
                )
                run_inference(
                    [model_name],  # One model at a times
                    special_id,
                    freeform_text,
                    normalizing,
                    model=model,
                    tokenizer=tokenizer,
                )
    else:
        freeform_text = ""
        visualization = True
        normalizing = False
        special_id = ""

        run_inference(
            MODEL_NAMES,
            special_id,
            freeform_text,
            normalizing,
        )

# %%
