import os
import openai
import pandas as pd
import json

import prompts

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.getenv('OPENAI_API_KEY')



def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]



def get_prompt_data(metadata_filename, dataset_filename, num_samples=20):
    with open(metadata_filename) as f:
        metadata = f.read()

    prompt = prompts.DATALOADER_PROMPT.format(metadata=metadata)
    response_json = get_completion(prompt)
    data = json.loads(response_json)


    df = pd.read_csv(dataset_filename)

    data['subset'] = df.sample(n=num_samples).values.tolist()

    return data



if __name__ == '__main__':
    data = get_prompt_data('data/adult.names', 'data/adult.data.csv')

    print(json.dumps(data, indent=4))