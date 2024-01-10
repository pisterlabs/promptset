import pickle
import pandas as pd
import os
import openai
import numpy as np
import ipdb
import re
from tqdm import tqdm
import time

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
import spacy
import scipy
from data_utils import *
from eval_utils import *

openai.api_key = os.environ['OPENAI_KEY']


def run_gpt3_ner_df(df, params):
    gpt3_output = []
    predictions = []
    final_prompts = []
    cost_list = []
    time_list = []

    sep = params['sep']

    for i, row in tqdm(df.iterrows()):
        start_time = time.time()
        prompt = row['test_ready_prompts']
        sent = row['sents']
        original_sent = row['orig_tok_sent']

        overall_instructions = params['overall_instructions'].strip()
        if overall_instructions != '':
            prompt = overall_instructions + '\n\n' + prompt

        final_prompts.append(prompt)

        logit_bias_sents = [sent, original_sent]
        sample = run_gpt3_ner(params['model'],
                              prompt,
                              logit_bias_sents,
                              params['max_tokens'],
                              sep,
                              logprobs=1,
                              fine_tuning=params['fine_tuning'],
                              logit_bias=params['logit_bias'])

        gpt3_output.append(sample)
        prediction = sample['choices'][0]['text']
        cost = sample['davinci_cost']
        # Lowercasing all predictions
        prediction = prediction.lower().strip()
        predictions.append(prediction)
        cost_list.append(cost)
        time_list.append(time.time() - start_time)


    df["predictions"] = [p.split(sep.strip()) for p in predictions]
    df['gpt3_output_predictions'] = gpt3_output
    df['final_input_prompts'] = final_prompts
    df['cost'] = cost_list
    df['time'] = time_list

    return df

from scipy.special import softmax

def compute_weight_matrix(samp, first_token_verbalized_labels):
    logprobs_per_cat = [samp["choices"][0]["logprobs"]["top_logprobs"][0][l] for l in first_token_verbalized_labels]
    mat = np.linalg.inv(np.diag(softmax(logprobs_per_cat)))
    return mat

def predict_calibrated_output(samp, weight_matrix, verbalized_labels, first_token_verbalized_labels):
    phat = [samp["choices"][0]["logprobs"]["top_logprobs"][0][l] for l in first_token_verbalized_labels]
    phat_mat = np.diag(softmax(phat))
    qhat = softmax(np.diag(np.matmul(weight_matrix, phat_mat)))
    pred = verbalized_labels[np.argmax(qhat)]
    return pred

def calibrated_output_probs(samp, weight_matrix, verbalized_labels, first_token_verbalized_labels):
    phat = [samp["choices"][0]["logprobs"]["top_logprobs"][0][l] for l in first_token_verbalized_labels]
    phat_mat = np.diag(softmax(phat))
    qhat = softmax(np.diag(np.matmul(weight_matrix, phat_mat)))
    return qhat

def run_gpt3_ner(engine,
                 prompt,
                 logit_bias_texts,
                 max_tokens,
                 sep,
                 logit_bias,
                 fine_tuning,
                 new_line_logit_bias=None,
                 logprobs=1,
                 logit_biases=None):

    if engine == 'cost':
        sample = {'choices': [{'text': ''}]}
    else:
        if logit_biases is None:
            logit_biases = {}
            logit_bias = int(logit_bias)

            for logit_bias_text in logit_bias_texts:
                # constrain potential output to tokens in test sentence using logit bias
                token_ids = tokenizer.encode(logit_bias_text)

                for i, token_id in enumerate(token_ids):
                    logit_biases[token_id] = logit_bias

                    token = tokenizer.decode(token_id)

                    if i == 0:
                        for space_token_id in tokenizer.encode(' ' + token):
                            if space_token_id not in logit_biases:
                                logit_biases[space_token_id] = logit_bias

                    for token_sep_id in tokenizer.encode(token + sep):
                        if token_sep_id not in logit_biases:
                            logit_biases[token_sep_id] = logit_bias

                    for token_nl_id in tokenizer.encode(token + '\n'):
                        if token_nl_id not in logit_biases:
                            logit_biases[token_nl_id] = logit_bias

            # Adding bias for separator
            sep_token = tokenizer.encode(sep)
            logit_biases[sep_token[0]] = logit_bias

            sep_token = tokenizer.encode(' ' +sep+ ' ')
            logit_biases[sep_token[0]] = logit_bias

            # Adding bias for newline (token id 198)
            if new_line_logit_bias is None:
                new_line_logit_bias = logit_bias

            logit_biases[198] = new_line_logit_bias

        if fine_tuning:
            #Adding fine tuning empty completion
            none_tokens = tokenizer.encode('None')
            for tok in none_tokens:
                logit_biases[tok] = logit_bias

            none_tokens = tokenizer.encode(' None ')
            for tok in none_tokens:
                logit_biases[tok] = logit_bias

        max_tokens = int(max_tokens)
        logprobs = int(logprobs)

        prompt = prompt.strip()

        # use API to generate completion
        if fine_tuning:
            sample = openai.Completion.create(model=engine,
                                              prompt=prompt,
                                              max_tokens=max_tokens,
                                              temperature=0.0,
                                              logit_bias=logit_biases,
                                              logprobs=logprobs,
                                              presence_penalty=-0.001,
                                              stop=["\n", "<|endoftext|>"])
        else:
            sample = openai.Completion.create(engine=engine,
                                              prompt=prompt,
                                              max_tokens=max_tokens,
                                              temperature=0.0,
                                              logit_bias=logit_biases,
                                              logprobs=logprobs,
                                              presence_penalty=-0.001,
                                              stop=["\n", "<|endoftext|>"])

        time.sleep(0.1)

    tokens = tokenizer.encode(prompt)
    cost_sample = {'tokens': tokens, 'num_tokens': len(tokens),
         'davinci_cost': 0.00006 * (len(tokens) + max_tokens)}

    sample.update(cost_sample)

    return sample


def run_gpt3_re_df(df, params):
    calibrate_outputs = []
    uncalibrated_predictions = []
    gpt3_output = []
    final_predictions = []
    weight_matrices = []
    final_prompts = []
    calibration_prompts = []
    cost_list = []
    time_list = []

    # ADD COLUMN OF TRAIN ONLY FOR CONTEXTUAL CALIBRATION
    for i, row in tqdm(df.iterrows()):
        start_time = time.time()

        prompt = row['test_ready_prompts']
        empty_prompt = row['empty_prompts']

        overall_instructions = params['overall_instructions'].strip()
        if overall_instructions != '':
            prompt = overall_instructions + '\n\n' + prompt

        label_verbalizer = params['label_verbalizer']
        label_list = np.sort(list(label_verbalizer.keys()),kind='stable')
        verbalized_labels = [label_verbalizer[l] for l in label_list]

        final_prompts.append(prompt)

        sample = run_gpt3_re(params['model'], prompt, verbalized_labels, params['fine_tuning'], params['max_tokens'])
        first_token_verbalized_labels = [tokenizer.decode(tokenizer.encode(" " + label)[0]) for label in verbalized_labels]

        uncalibrated_probs = [sample["choices"][0]["logprobs"]["top_logprobs"][0][l] for l in first_token_verbalized_labels]
        uncalibrated_prediction = verbalized_labels[np.argmax(uncalibrated_probs)]

        uncalibrated_predictions.append(uncalibrated_prediction)
        gpt3_output.append(sample)

        if params['calibration']:
            few_shot_example = prompt[:-1 * len(empty_prompt)] + params['sent_intro'] + ' N/A\n' + params['retrieval_message'].format('N/A', 'N/A')
            calibration_prompts.append(few_shot_example)
            calibrate_sample = run_gpt3_re(params['model'], few_shot_example, verbalized_labels, params['fine_tuning'], params['max_tokens'])

            weight_matrix = compute_weight_matrix(calibrate_sample, first_token_verbalized_labels)
            final_prediction = predict_calibrated_output(sample, weight_matrix, verbalized_labels, first_token_verbalized_labels)
            weight_matrix = np.diag(weight_matrix)
        else:
            weight_matrix = None
            final_prediction = uncalibrated_prediction
            calibrate_sample = None
            calibration_prompts.append(None)

        weight_matrices.append(weight_matrix)
        calibrate_outputs.append(calibrate_sample)
        final_predictions.append(final_prediction)

        cost = sample['davinci_cost']
        cost_list.append(cost)
        time_list.append(time.time() - start_time)


    df["predictions"] = final_predictions
    df['uncalibrated_predictions'] = uncalibrated_predictions
    df['gpt3_output_predictions'] = gpt3_output
    df['gpt3_output_predictions_calibrate_prompt'] = calibrate_outputs
    df['weight_matrices'] = weight_matrices
    df['final_input_prompts'] = final_prompts
    df['final_calibration_prompts'] = calibration_prompts
    df['cost'] = cost_list
    df['time'] = time_list

    return df

def run_gpt3_re(engine, prompt, classes, fine_tuning, max_tokens=1, logit_bias=100):

    tokens = tokenizer.encode(prompt)

    if len(tokens) > 2049:
        ipdb.set_trace()
    if engine == 'cost':
        sample = {'choices': [{'text': ''}]}
    else:
        logit_biases = {}
        for label in classes:
            label = tokenizer.encode(" " + label)[0]
            logit_biases[label] = logit_bias

        prompt = prompt.strip()

        # use API to generate completion
        if fine_tuning:
            sample = openai.Completion.create(model=engine,
                                              prompt=prompt,
                                              max_tokens=int(max_tokens),
                                              temperature=0,
                                              logit_bias=logit_biases,
                                              stop=["\n", "<|endoftext|>"],
                                              logprobs=len(classes))
        else:
            sample = openai.Completion.create(engine=engine,
                                              prompt=prompt,
                                              max_tokens=int(max_tokens),
                                              temperature=0,
                                              logit_bias=logit_biases,
                                              stop=["\n", "<|endoftext|>"],
                                              logprobs=len(classes))
        time.sleep(0.1)

    cost_sample = {'tokens': tokens, 'num_tokens': len(tokens),
         'davinci_cost': 0.00006 * (len(tokens) + max_tokens)}

    sample.update(cost_sample)

    return sample