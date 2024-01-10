"""Code for GPT-3"""


import openai
import numpy as np
import pandas as pd

from tqdm import tqdm

## 
openai.organization = None
openai.api_key = None


def log_odds_gpt3(experimental_sentence,word_idx): 
    """Get log_odds of start/end word in sentence."""
    output = openai.Completion.create(
        engine = "ada",
        prompt = experimental_sentence,
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
    logprob = output.to_dict()['choices'][0].to_dict()['logprobs']["token_logprobs"][word_idx]
    target_word = output.to_dict()['choices'][0].to_dict()['logprobs']["tokens"][word_idx][1:]

    return logprob


def main(filename):
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
    probs = []
    beliefs = []
    consistency = []
    with tqdm(total=df_passages.shape[0]) as pbar:            
        for index, row in df_passages.iterrows():

            # conditions
            belief, con = row['Condition code'].split("-")
            beliefs.append(belief)
            consistency.append(con)

            # Get passage
            text = row['Scenario']
            # Get log odds for these words
            p = log_odds_gpt3(text, -3)
            # Add ratio to log-odds list
            probs.append(p)

            pbar.update(1)

    # Add to dataframe
    df_passages['log_prob'] = probs
    df_passages['belief'] = beliefs
    df_passages['consistency'] = consistency

    # Save file
    df_passages.to_csv("data/processed/{TASK}_gpt3_surprisals.csv".format(TASK=filename))


if __name__ == "__main__":
    from argparse import ArgumentParser 

    parser = ArgumentParser()

    parser.add_argument("--path", type=str, dest="filename",
                        default="bradford-fb")
    
    args = vars(parser.parse_args())
    print(args)
    main(**args)