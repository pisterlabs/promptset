


import os
import openai
from openai.embeddings_utils import cosine_similarity, get_embedding

import sys
import pandas as pd
import numpy as np
from tqdm import tqdm


import random
import argparse

import paths
import config


from utils import make_and_clear


from config import CHAT_RPM_LIMIT, CHAT_TPM_LIMIT



    
def main(args):


    destination = args.destination
    make_and_clear(destination)



    df = pd.read_csv(args.synthetic_prompts)


    print(df)


    if args.num_prompts == -1:
        idx = list(range(len(df)))
    else:
        idx = np.round(np.linspace(0, len(df) - 1, args.num_prompts)).astype(int)


    df_temp = df.iloc[idx]

    print(df_temp)

    if True:

        # Load your API key from an environment variable or secret management service
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        prompt = df_temp[config.PROMPT].tolist()


        answer = input(f'''Confirm transmission of {len(prompt)} prompts. Enter yes or no: ''') 

        if answer == 'yes':


            # completion = openai.Completion.create( \
            #     model = args.model, 
            #     prompt = prompt,
            #     temperature = args.temperature, 
            #     max_tokens = args.max_tokens
            #     )


            # responses = completion['choices']
            # responses = [r['text'] for r in responses]
            # responses = [r.lstrip() for r in responses]






            model = "gpt-3.5-turbo-0301"
            responses = []

            pbar = tqdm(total=len(prompt))
            for p in prompt:
                messages = [{'role': 'user', 'content': p}]

                completion = openai.ChatCompletion.create( \
                    model = model, 
                    messages = messages,
                    temperature = args.temperature, 
                    # max_tokens = max_tokens
                    )

                response = completion['choices'][0]['message']['content'].lstrip()
                responses.append(response)
                pbar.update(1)
            pbar.close()


            df_temp[config.RESPONSE] = responses
            df_temp.to_csv(args.synthetic_responses)


        else:
            print('Did not transmit')


if __name__ == '__main__':


    destination = paths.process_synthetic_prompts
    synthetic_prompts = paths.synthetic_prompts
    synthetic_responses = paths.synthetic_responses


    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('--destination', type=str, default=destination, help="output directory")
    arg_parser.add_argument('--synthetic_prompts', type=str, default=synthetic_prompts, help="output directory")
    arg_parser.add_argument('--synthetic_responses', type=str, default=synthetic_responses, help="output directory")    
    arg_parser.add_argument('--num_prompts', type=int, default=20, help="output directory")
    arg_parser.add_argument('--model', type=str, default='text-curie-001', help="openai lm")
    arg_parser.add_argument('--temperature', type=float, default=1.0, help="model temperature")
    arg_parser.add_argument('--max_tokens', type=int, default=300, help="max number of generated tokens")

    args, _ = arg_parser.parse_known_args()

    sys.exit(main(args)) 

    args, _ = arg_parser.parse_known_args()

    sys.exit(main(args)) 