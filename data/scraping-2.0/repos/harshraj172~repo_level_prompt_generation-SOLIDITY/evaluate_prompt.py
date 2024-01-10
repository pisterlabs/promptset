import os, json
import openai 
import pandas as pd
import time 
from tqdm.auto import tqdm
import argparse

def gen_text(prompt):
    response = openai.Completion.create(
      engine="code-davinci-002",
      prompt=prompt,
      stop='\n',
      temperature=0.5,
      max_tokens=20,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    return response['choices'][0]['text']

def setup_args():
    """
    Description: Takes in the command-line arguments from user
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir_path", type=str, help="directory where the prompt data is stored")
    parser.add_argument("--openai_api_key", type=str, help="openai api key")

    return parser.parse_args()

if __name__ == '__main__':

    args = setup_args()

    os.environ["OPENAI_API_KEY"]=args.openai_api_key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # prompts
    for dir_ in tqdm(os.listdir(f"{args.dir_path}/rlpg")):
        rlpg_outs, rlpg_ptypes, target_holes = [], [], []
        for file in os.listdir(f"{args.dir_path}/rlpg/{dir_}"):
            with open(f"{args.dir_path}/rlpg/{dir_}/{file}", 'r') as f:
                rlpg_prompt = f.read().rstrip()
            rlpg_out = gen_text(rlpg_prompt)
            rlpg_outs.append(rlpg_out)
            rlpg_ptypes.append(str(file).split('.')[0])
            
            with open(f"{'/'.join(args.dir_path.split('/')[:-1])}/meta/{dir_}.json", 'r') as f:
                dict_lst = json.load(f)
            target_holes.append(dict_lst[1]['target_hole'])

        with open(f"{args.dir_path}/normal/{dir_}.txt", 'r') as f:
            default_prompt = f.read().rstrip()
        default_out = gen_text(default_prompt)
        default_outs = [default_out]*len(rlpg_outs)
        

        temp_df = pd.DataFrame({
            'rlpg_prompt-type': rlpg_ptypes,
            'rlpg_output' : rlpg_outs,
            'default_output': default_outs,
            'target-hole': target_holes,
        })
        
        try:
            df = pd.concat([df, temp_df], axis=0)
        except:
            df = temp_df
        df.to_csv("eval-rlpg_prompt.csv", index=False)
        
