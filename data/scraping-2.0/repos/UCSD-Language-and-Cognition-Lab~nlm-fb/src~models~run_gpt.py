"""Code for GPT-3"""


import openai
import time
import numpy as np
import pandas as pd

from tqdm import tqdm

## Read in key
with open('src/models/gpt_key', 'r') as f:
    lines = f.read().split("\n")

org = lines[0]
api_key = lines[1]
openai.organization = org # org
openai.api_key = api_key # api_key


def log_odds_gpt3(experimental_sentence,start, end, word_idx, model="ada"): 
    """Get log_odds of start/end word in sentence."""
    v_start = experimental_sentence.replace("[MASK].", start)
    v_end = experimental_sentence.replace("[MASK].", end)
    start_output = openai.Completion.create(
        engine = model,
        prompt = v_start,
        max_tokens = 0,
        temperature = 0,
        n = 1,
        stream = False,
        logprobs = 0,
        stop = "\n",
        echo = True
        )

    end_output = openai.Completion.create(
        engine = model,
        prompt = v_end,
        max_tokens = 0,
        temperature = 0,
        n = 1,
        stream = False,
        logprobs = 0,
        stop = "\n",
        echo = True
        )

    ### For cupboard and toolbox, take product of tool + box and cup + board.
    if start in ['cupboard', 'toolbox']:
        start_logprobs = start_output.to_dict()['choices'][0].to_dict()['logprobs']["token_logprobs"][word_idx-1:]
        start_target_words = start_output.to_dict()['choices'][0].to_dict()['logprobs']["tokens"][word_idx-1:]

        start_target_word = ''.join(start_target_words).replace(" ", "")
        start_logprob = sum(start_logprobs)
    else:
        start_logprob = start_output.to_dict()['choices'][0].to_dict()['logprobs']["token_logprobs"][word_idx]
        start_target_word = start_output.to_dict()['choices'][0].to_dict()['logprobs']["tokens"][word_idx].replace(" ", "")


    # info for start
    end_logprob = end_output.to_dict()['choices'][0].to_dict()['logprobs']["token_logprobs"][word_idx]
    end_target_word = end_output.to_dict()['choices'][0].to_dict()['logprobs']["tokens"][word_idx].replace(" ", "")

    # ratio
    log_odds = start_logprob - end_logprob

    lp_pred = start_target_word if log_odds > 0 else end_target_word 

    return {'log_odds': log_odds,
            'token_c1': start_target_word,
            'end_logprob': end_logprob,
            'start_logprob': start_logprob,
            'passage': experimental_sentence,
            'token_c2': end_target_word,
            "lp_pred": lp_pred}


def pred_tokens(prompt, n=1, model="ada"): 
    """Get log_odds of start/end word in sentence."""
    prompt = prompt.replace(" [MASK].", "")

    output = openai.Completion.create(
        engine = model,
        prompt = prompt,
        max_tokens = n,
        temperature = 0,
        n = 1,
        logprobs = 1,
        echo = False
        )

    tokens = output.to_dict()['choices'][0].to_dict()["logprobs"]["tokens"]
    token_logprobs = output.to_dict()['choices'][0].to_dict()["logprobs"]["token_logprobs"]

    return (tokens, token_logprobs)


def main(filename, c1, c2, model='ada'):
    """
    Run GPT-3 on stims. 

    Requires that filename point to a .csv file with a "passage" column, as well as 
    columns for the two words (c1 vs. c2) to be comparing (e.g., start vs. end).

    """

    # Assemble data path
    data_path = "data/stims/{FILE}.csv".format(FILE=filename)

    # Usnig which model?
    print("Using model: {M}".format(M = model))
    
    # Read in stims
    df_passages = pd.read_csv(data_path)
    print("#passages: {M}".format(M = len(df_passages)))

    # Remove any line splits
    df_passages['passage'] = df_passages['passage'].apply(lambda x: x.replace("\n", ""))
    df_passages['passage'] = df_passages['passage'].apply(lambda x: x.replace("[MASK]. ", "[MASK]."))

    # For each passage, get log-odds of c1 vs. c2
    results = []
    with tqdm(total=df_passages.shape[0]) as pbar:            
        for index, row in df_passages.iterrows():

            # Get passage
            text = row['passage']
            # Get substitutions for masked word
            candidates = [row[c1], row[c2]]
            # Get log odds for these words
            info = log_odds_gpt3(text, row[c1], row[c2], -1, model=model)

            # Get predictions from model
            tokens, token_logprobs = pred_tokens(text, n=2, model=model)
            # Add to info dict
            info["pred_t1"] = tokens[0].strip()
            info["pred_t2"] = tokens[1].strip()
            info["pred_lp1"] = token_logprobs[0]
            info["pred_lp2"] = token_logprobs[1]

            # Add ratio to log-odds list
            results.append(info)

            ## Pause
            if index == 250:
                time.sleep(1)
            elif index == 500:
                time.sleep(1)
            elif index == 750:
                time.sleep(1)

            pbar.update(1)

    # Create dataframe
    df_results = pd.DataFrame(results)
    # Add to dataframe
    df_passages = pd.merge(df_passages, df_results, on = "passage")

    # Save file
    df_passages.to_csv("data/processed/{TASK}_gpt3-{m}_surprisals_probs.csv".format(TASK=filename, m=model))


if __name__ == "__main__":
    from argparse import ArgumentParser 

    parser = ArgumentParser()

    parser.add_argument("--path", type=str, dest="filename",
                        default="fb")
    parser.add_argument("--c1", type=str, dest="c1",
                        default="start")
    parser.add_argument("--c2", type=str, dest="c2",
                        default="end")
    parser.add_argument("--m", type=str, dest="model",
                        default="davinci")
    
    args = vars(parser.parse_args())
    print(args)
    main(**args)