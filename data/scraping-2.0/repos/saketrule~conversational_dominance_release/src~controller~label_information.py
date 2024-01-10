# Imports
import re
import os
import sys
from typing import Any
import numpy as np
import pandas as pd
import logging
from scriptify import scriptify
from os.path import dirname, abspath

repo_path = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(repo_path)

from src.agents.openai_agent import OpenAIAgent

logging.basicConfig(level=logging.INFO)

utterances_per_prompt = 8
multisimo_path = os.path.join(repo_path, 'data/processed/multisimo/transcript_dominance.csv')
gpt4_student_dom_path = os.path.join(repo_path, 'data/external/gpt4_dataset/student_dominated.txt')
output_folder = os.path.join(repo_path, "data/results/")


class GPTData:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dominant_speaker = "Student" if "student_dominated" in data_path.split("/")[-1] else "Teacher"
        self.df = None
    
    def load_data(self):
        with open(self.data_path, 'r') as f:
            transcripts = f.read().split('\n\n')
        
        transcripts = [x for x in transcripts if x != '']

        df_data = []
        for transcript in transcripts:
            # Get utterances
            utterances = transcript.split("\n")
            speakers = [x.split(":")[0] for x in utterances[:-1]]
            dominant_speaker = self.dominant_speaker
            df_data.append([transcript, utterances, speakers, dominant_speaker])
            
        df_transcripts = pd.DataFrame(df_data, columns=["transcript", "utterances", "speakers", "dominant_speaker"])

        return df_transcripts


class MultisimoData:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
    
    def load_data(self):
        multisimo_df = pd.read_csv(self.data_path)
        df_data = []
        for el,path in zip(multisimo_df["file_content"], multisimo_df["file_name"]):
            dataset = re.sub(r'\[', r'\n[', el).split("\n")[1:]
            pattern = r'\[(SPK\d|MOD)\]'
            matches = re.findall(pattern, "".join(dataset))
            speakers= np.unique(matches)
            dominant_speaker = None
            df_data.append([el, dataset, matches, dominant_speaker])

        df_transcripts = pd.DataFrame(df_data, columns=["transcript", "utterances", "speakers", "dominant_speaker"])
        
        return df_transcripts


class LabelTranscriptDf:

    prompt_template = """For each of the following utterance, Is the utterance information giving, information seeking or neutral? Respond with {n_utterances} lines with only "Information giving", "Information seeking" or "Neutral" on every line -

{utterances}
"""

    def __init__(self) -> None:
        self.agent = OpenAIAgent()
    

    def get_prompts(self, utterances, n_utterances):
        
        for i in range(0,len(utterances),n_utterances):
            yield self.prompt_template.format(
                n_utterances=n_utterances,
                utterances="\n".join(utterances[i:i+n_utterances])
            )

    def __call__(self, df_transcripts, utterances_per_prompt=7, mock=False) -> Any:
        if "response" not in df_transcripts.columns:
            df_transcripts["response"] = ""
            df_transcripts["utterances_clean"] = [[] for _ in range(len(df_transcripts))]
        
        logging.debug(len(df_transcripts))
        for i, row in df_transcripts.iterrows():
            utterances_clean = []
            for x in row["utterances"]:
                x = x.split("]")
                assert(len(x)==2)
                if len(x) > 1:
                    x = f'{x[0]}]: "{x[1].strip()}"'
                    utterances_clean.append(x)

            
            prompts = list(self.get_prompts(utterances_clean, utterances_per_prompt))
            logging.info("Number of prompts = {}".format(len(prompts)))
            prompts_str = '\n'.join(prompts)
            logging.debug(f"\n[Prompts {i}] -\n{prompts_str}\n")

            if not mock:
                resp_texts = []
                for i_p, prompt in enumerate(prompts):
                    logging.info(f"Prompt {i_p}/{len(prompts)} - {prompt}")
                    resp = self.agent.get_response(prompt)
                    resp_texts.append(self.agent.response_text_content(resp))
                    logging.info(f"Response {i_p}/{len(prompts)} - \n{resp_texts[-1]}\n")
                resp_text = '\n'.join(resp_texts)
                
            else:
                resp_text = "mock"
            
            print(f"I={i}")
            df_transcripts.loc[i, "response"] = resp_text
            df_transcripts.at[i, "utterances_clean"] = utterances_clean

        # Split text in response into utterances
        df_transcripts["response_utterances"] = df_transcripts["response"].apply(lambda x: x.split("\n"))
        df_transcripts["n_responses_match"] = df_transcripts.apply(lambda x: len(x["utterances_clean"]) == len(x["response_utterances"]), axis=1)

        return df_transcripts

def print_df_sample(df):
    logging.info(f"Shape: {df.shape}")
    # Log Transpose of one row
    logging.info(f"Sample Row:\n{df.iloc[0].T}")

def main():
    """
    Read Data
    """
    ## GPT4 Data
    # data_loader = GPTData(gpt4_student_dom_path)
    # df_transcripts = data_loader.load_data()

    ## Multisimo Data
    data_loader = MultisimoData(multisimo_path)
    df_transcripts = data_loader.load_data()

    print_df_sample(df_transcripts)

    label_agent = LabelTranscriptDf()
    df_transcripts = label_agent(df_transcripts.iloc[:5], utterances_per_prompt=utterances_per_prompt, mock=False)

    output_path = os.path.join(output_folder, 'multisimo_labels.csv')
    print_df_sample(df_transcripts)
    df_transcripts.to_csv(output_path, index=False)



if __name__ == "__main__":
    main()
