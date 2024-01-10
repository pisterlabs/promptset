"""Code for GPT-3"""


import openai
import numpy as np
import pandas as pd

from tqdm import tqdm

## 
openai.organization = None
openai.api_key = None


def log_odds_gpt3(experimental_sentence,start, end, word_idx): 
    """Get log_odds of start/end word in sentence."""
    v_start = experimental_sentence.replace("[MASK]", start)
    v_end = experimental_sentence.replace("[MASK]", end)
    start_output = openai.Completion.create(
        engine = "ada",
        prompt = v_start,
        max_tokens = 1,
        temperature = 1,
        top_p = 1,
        n = 1,
        stream = False,
        logprobs = 1,
        stop = "\n",
        echo = True
        )

    end_output = openai.Completion.create(
        engine = "ada",
        prompt = v_end,
        max_tokens = 1,
        temperature = 1,
        top_p = 1,
        n = 1,
        stream = False,
        logprobs = 1,
        stop = "\n",
        echo = True
        )

    # info for start
    start_logprob = start_output.to_dict()['choices'][0].to_dict()['logprobs']["token_logprobs"][word_idx]
    start_target_word = start_output.to_dict()['choices'][0].to_dict()['logprobs']["tokens"][word_idx][1:]

    # info for start
    end_logprob = end_output.to_dict()['choices'][0].to_dict()['logprobs']["token_logprobs"][word_idx]
    end_target_word = end_output.to_dict()['choices'][0].to_dict()['logprobs']["tokens"][word_idx][1:]

    # ratio
    log_odds = start_logprob - end_logprob

    return log_odds


def main(filename, c1, c2):
    """
    Run GPT-3 on stims. 

    Requires that filename point to a .csv file with a "passage" column, as well as 
    columns for the two words (c1 vs. c2) to be comparing (e.g., start vs. end).

    (#TODO: This main function could/should probabably be generalized to encompass other models too.)
    """

    # Assemble data path
    data_path = "data/stims/{FILE}.csv".format(FILE=filename)
    
    # Read in stims
    df_passages = pd.read_csv(data_path)

    # For each passage, get log-odds of c1 vs. c2
    odds = []
    with tqdm(total=df_passages.shape[0]) as pbar:            
        for index, row in df_passages.iterrows():

            # Get passage
            text = row['passage']
            # Get substitutions for masked word
            candidates = [row[c1], row[c2]]
            # Get log odds for these words
            ratio = log_odds_gpt3(text, row[c1], row[c2], -3)
            # Add ratio to log-odds list
            odds.append(ratio)

            pbar.update(1)

    # Add to dataframe
    df_passages['log_odds'] = odds

    # Save file
    df_passages.to_csv("data/processed/{TASK}_gpt3_surprisals.csv".format(TASK=filename))


if __name__ == "__main__":
    from argparse import ArgumentParser 

    parser = ArgumentParser()

    parser.add_argument("--path", type=str, dest="filename",
                        default="fb")
    parser.add_argument("--c1", type=str, dest="c1",
                        default="start")
    parser.add_argument("--c2", type=str, dest="c2",
                        default="end")
    
    args = vars(parser.parse_args())
    print(args)
    main(**args)